# app.py - DLR-PCVW Acne Severity Classification Web App
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import requests
from io import BytesIO
import time
import random

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="DLR-PCVW - Acne Severity Classification",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - Minimalist Medical + Subtle Glassmorphism
# ============================================================
def load_css():
    """Load custom CSS with minimalist medical + subtle glassmorphism"""
    st.markdown("""
    <style>
    /* Clean background */
    .stApp {
        background: #0b0f14;
    }
    
    /* Subtle glassmorphism for main containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
    }
    
    /* Compact Header with subtle glass */
    .main-header {
        text-align: center;
        padding: 0.8rem 0 0.5rem 0;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #1a2332;
        letter-spacing: -0.5px;
        margin: 0;
        padding: 0;
    }
    
    .main-header h1 .accent {
        color: #3182ce;
    }
    
    .main-header .subtitle {
        font-size: 0.9rem;
        color: #4a5568;
        margin-top: 0.1rem;
        font-weight: 400;
    }
    
    .main-header .badge {
        display: inline-block;
        background: #3182ce;
        color: white;
        padding: 0.15rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    /* Severity Card with subtle glass */
    .severity-card {
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.3rem 0;
        border: 2px solid #e2e8f0;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .severity-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    
    .severity-card .grade {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    
    .severity-card .sub {
        font-size: 0.9rem;
        color: #718096;
        margin-top: 0.15rem;
    }
    
    .confidence-bar {
        margin-top: 0.5rem;
        background: #edf2f7;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
    }
    
    .confidence-bar .fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease;
    }
    
    /* Metric cards with subtle glass */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(226, 232, 240, 0.6);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a2332;
    }
    
    .metric-card .label {
        font-size: 0.75rem;
        color: #718096;
        margin-top: 0.1rem;
    }
    
    .metric-row {
        display: flex;
        gap: 0.5rem;
        margin: 0.3rem 0;
    }
    
    .metric-row .metric-card {
        flex: 1;
    }
    
    /* Disclaimer */
    .disclaimer {
        font-size: 0.8rem;
        color: #718096;
        padding: 0.8rem;
        background: rgba(247, 250, 252, 0.9);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-top: 0.8rem;
    }
    
    /* Model Pipeline with glass */
    .model-pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
        padding: 0.6rem;
        background: rgba(247, 250, 252, 0.85);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 12px;
        flex-wrap: wrap;
        font-size: 0.7rem;
        color: #4a5568;
        border: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    .model-pipeline .step {
        background: rgba(255, 255, 255, 0.8);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
        font-size: 0.7rem;
    }
    
    .model-pipeline .arrow {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1a2332;
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #2d3748;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26, 35, 50, 0.15);
    }
    
    /* Sidebar with glass */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #cbd5e0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(247, 250, 252, 0.5);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    /* Progress animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .processing-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(226, 232, 240, 0.6);
        margin: 0.5rem 0;
    }
    
    /* Prototype similarity bar container */
    .similarity-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid rgba(226, 232, 240, 0.5);
        margin: 0.3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================================
# MODEL DEFINITION
# ============================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        self.features.eval()
        return self.features(x)

class DLR_Pool(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=256, n_classes=4):
        super().__init__()
        self.stat_proj = nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim*2)
        )
        self.lesion_scorer = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.var_weight = nn.Parameter(torch.full((in_dim,), -3.0))
        self.head_predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))
            for _ in range(4)
        ])

    def forward(self, fm, return_vis=False):
        B, C, H, W = fm.shape
        stats = torch.cat([fm.mean(dim=[2,3]), fm.var(dim=[2,3])], dim=1)
        film = self.stat_proj(stats)
        gamma = film[:, :C].view(B, C, 1, 1)
        beta = film[:, C:].view(B, C, 1, 1)
        fm_calib = fm * (1 + gamma) + beta
        tokens = fm_calib.view(B, C, -1).permute(0, 2, 1)
        scores = torch.sigmoid(self.lesion_scorer(tokens))
        score_sum = scores.sum(dim=1) + 1e-6
        mu = (tokens * scores).sum(dim=1) / score_sum
        diff_sq = (tokens - mu.unsqueeze(1)) ** 2
        var = (diff_sq * scores).sum(dim=1) / score_sum
        weight = F.softplus(self.var_weight)
        feat = mu + weight * var
        feat = F.normalize(feat, p=2, dim=1)
        head_logits = [head(feat) for head in self.head_predictors]
        
        if return_vis:
            return feat, head_logits, scores.view(B, -1), mu, var, weight
        return feat, head_logits

class DLR_ProtoNet(nn.Module):
    def __init__(self, feat_dim=1280, n_classes=4):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.dlr_pool = DLR_Pool(feat_dim, 256, n_classes)
        self.register_buffer('running_mean', torch.zeros(n_classes, feat_dim))
        self.register_buffer('running_count', torch.zeros(n_classes))
        self.semantic_step = nn.Parameter(torch.zeros(n_classes, feat_dim))

    def calibrate(self, raw_protos, sy):
        classes = torch.unique(sy, sorted=True)
        cal = []
        for ny, c in enumerate(classes):
            gc = c.item()
            raw = raw_protos[ny]
            if self.running_count[gc] > 5:
                cal.append(0.75 * raw + 0.25 * self.running_mean[gc])
            else:
                cal.append(raw)
        return torch.stack(cal)

    def forward(self, sx, sy, qx, training=False, return_vis=False):
        s_fm = self.extractor(sx)
        q_fm = self.extractor(qx)
        
        if return_vis:
            s_feat, s_heads, s_scores, s_mu, s_var, s_weight = self.dlr_pool(s_fm, return_vis=True)
            q_feat, q_heads, q_scores, q_mu, q_var, q_weight = self.dlr_pool(q_fm, return_vis=True)
        else:
            s_feat, s_heads = self.dlr_pool(s_fm)
            q_feat, q_heads = self.dlr_pool(q_fm)
        
        classes = torch.unique(sy, sorted=True)
        raw = torch.stack([s_feat[sy == c].mean(0) for c in classes])
        cal = self.calibrate(raw, sy)
        final = cal + 0.05 * F.normalize(self.semantic_step[classes], dim=1)
        logits = F.cosine_similarity(q_feat.unsqueeze(1), final.unsqueeze(0), dim=2)
        logits = logits * 16.0
        
        if return_vis:
            return logits, final, s_heads, {
                'scores': q_scores[0].detach().cpu().numpy(),
                'mu': q_mu[0].detach().cpu().numpy(),
                'var': q_var[0].detach().cpu().numpy(),
                'weight': q_weight.detach().cpu().numpy()
            }
        return logits, final, None

# ============================================================
# PLOT FUNCTION
# ============================================================
def plot_prototype_similarity(probs, severity_names, severity_colors, pred_idx):
    """Plot prototype similarity bar chart"""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    bars = ax.bar(severity_names, probs, color=severity_colors, alpha=0.7, edgecolor='white', linewidth=1)
    
    # Highlight predicted class
    bars[pred_idx].set_alpha(1.0)
    bars[pred_idx].set_linewidth(2)
    bars[pred_idx].set_edgecolor('#1a2332')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Similarity', fontsize=9)
    ax.set_xlabel('Prototype', fontsize=9)
    ax.set_title('Prototype Similarity Scores', fontsize=10, fontweight='600')
    ax.grid(True, alpha=0.2, axis='y')
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.2%}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

# ============================================================
# PREDICTOR CLASS
# ============================================================
class DLRPredictor:
    def __init__(self, model_path_or_url, device='cpu'):
        self.device = device
        self.severity_names = ['Grade 0 (Clear)', 'Grade 1 (Mild)', 'Grade 2 (Moderate)', 'Grade 3 (Severe)']
        self.severity_colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
        self.severity_short = ['Clear', 'Mild', 'Moderate', 'Severe']
        self.models = []
        self.support_x = None
        self.support_y = None
        
        try:
            if model_path_or_url.startswith('http'):
                with st.spinner("Downloading model from Hugging Face..."):
                    response = requests.get(model_path_or_url)
                    checkpoint = torch.load(
                        BytesIO(response.content), 
                        map_location=self.device,
                        weights_only=False
                    )
            else:
                checkpoint = torch.load(
                    model_path_or_url, 
                    map_location=self.device,
                    weights_only=False
                )
            
            model_states = checkpoint['models']
            for state_dict in model_states:
                model = DLR_ProtoNet().to(self.device)
                model.load_state_dict(state_dict)
                model.eval()
                self.models.append(model)
            
            st.success(f"✅ Loaded {len(self.models)} ensemble models")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise
    
    def load_support_set(self, support_x, support_y):
        self.support_x = support_x.to(self.device)
        self.support_y = support_y.to(self.device)
    
    def get_support_subset(self, k_shot, num_classes=4):
        if self.support_x is None:
            raise ValueError("Support set not loaded!")
        
        support_x_list = []
        support_y_list = []
        
        for c in range(num_classes):
            mask = self.support_y == c
            class_x = self.support_x[mask]
            class_y = self.support_y[mask]
            
            if len(class_x) >= k_shot:
                indices = torch.randperm(len(class_x))[:k_shot]
                support_x_list.append(class_x[indices])
                support_y_list.append(class_y[indices])
            else:
                indices = torch.randint(0, len(class_x), (k_shot,))
                support_x_list.append(class_x[indices])
                support_y_list.append(class_y[indices])
        
        return torch.cat(support_x_list), torch.cat(support_y_list)
    
    @torch.no_grad()
    def predict(self, image_tensor, k_shot=5):
        if self.support_x is None:
            raise ValueError("Support set not loaded!")
        
        sx, sy = self.get_support_subset(k_shot)
        sx = sx.to(self.device)
        sy = sy.to(self.device)
        qx = image_tensor.unsqueeze(0).to(self.device)
        
        all_logits = []
        all_vis_data = []
        
        for model in self.models:
            logits, _, _, vis_data = model(sx, sy, qx, training=False, return_vis=True)
            all_logits.append(logits.cpu())
            all_vis_data.append(vis_data)
        
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probs = F.softmax(avg_logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        
        return {
            'prediction': pred.item(),
            'severity_name': self.severity_names[pred.item()],
            'severity_short': self.severity_short[pred.item()],
            'confidence': confidence.item(),
            'probabilities': probs.numpy().flatten(),
            'vis_data': all_vis_data[0],
            'k_shot': k_shot
        }

# ============================================================
# SUPPORT SET LOADING - FROM HUGGING FACE
# ============================================================
def load_support_set():
    """Load support set from Hugging Face or local file"""
    
    # First, check if support set exists locally (for faster loading)
    local_support_file = "support_set/support_set.pt"
    
    if os.path.exists(local_support_file):
        st.info("📁 Using local support set")
        data = torch.load(local_support_file, map_location='cpu', weights_only=False)
        return data['x'], data['y']
    
    # If not local, download from Hugging Face
    try:
        st.info("📥 Downloading support set from Hugging Face...")
        support_url = "https://huggingface.co/iamthearafatkhan/dlr-pcvw-acne-severity/resolve/main/support_set/support_set.pt"
        
        response = requests.get(support_url)
        if response.status_code == 200:
            # Load from memory
            data = torch.load(BytesIO(response.content), map_location='cpu', weights_only=False)
            
            # Optionally save locally for next time
            os.makedirs(os.path.dirname(local_support_file), exist_ok=True)
            torch.save(data, local_support_file)
            st.success(f"✅ Support set downloaded and cached locally")
            
            return data['x'], data['y']
        else:
            st.error(f"Failed to download support set (Status: {response.status_code})")
            # Fallback to random data
            return create_fallback_support_set()
            
    except Exception as e:
        st.error(f"Error loading support set: {e}")
        return create_fallback_support_set()

def create_fallback_support_set():
    """Create fallback support set if download fails"""
    st.warning("Using fallback support set. This may affect prediction accuracy.")
    support_x = torch.randn(20, 3, 224, 224)
    support_y = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3])
    return support_x, support_y

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_attention_overlay(image_tensor, attention_map):
    try:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if image_tensor.dim() == 3:
            img_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 4:
            img_tensor = image_tensor[:1]
        else:
            img_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        
        img_tensor = img_tensor.cpu()
        img_tensor = img_tensor * std + mean
        img_tensor = img_tensor.clamp(0, 1)
        img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        
        if isinstance(attention_map, np.ndarray):
            attention_tensor = torch.from_numpy(attention_map).float()
        else:
            attention_tensor = attention_map.float()
        
        if attention_tensor.dim() == 1:
            if attention_tensor.numel() == 49:
                attention_tensor = attention_tensor.reshape(7, 7)
            else:
                size = int(np.sqrt(attention_tensor.numel()))
                if size * size == attention_tensor.numel():
                    attention_tensor = attention_tensor.reshape(size, size)
                else:
                    size = int(np.ceil(np.sqrt(attention_tensor.numel())))
                    padded = torch.zeros(size * size)
                    padded[:attention_tensor.numel()] = attention_tensor
                    attention_tensor = padded.reshape(size, size)
        elif attention_tensor.dim() == 3:
            attention_tensor = attention_tensor.mean(dim=0)
        
        min_val = attention_tensor.min()
        max_val = attention_tensor.max()
        if max_val - min_val > 1e-8:
            attention_tensor = (attention_tensor - min_val) / (max_val - min_val)
        else:
            attention_tensor = torch.zeros_like(attention_tensor)
        
        attention_upsampled = F.interpolate(
            attention_tensor.unsqueeze(0).unsqueeze(0),
            size=(224, 224), 
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        min_val = attention_upsampled.min()
        max_val = attention_upsampled.max()
        if max_val - min_val > 1e-8:
            attention_upsampled = (attention_upsampled - min_val) / (max_val - min_val)
        else:
            attention_upsampled = np.zeros_like(attention_upsampled)
        
        heatmap = plt.cm.jet(attention_upsampled)[:, :, :3]
        overlay = 0.6 * img + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        return img, overlay, attention_upsampled
        
    except Exception as e:
        st.warning(f"Could not create attention overlay: {e}")
        img = np.zeros((224, 224, 3))
        return img, img, np.zeros((224, 224))

# ============================================================
def show_processing_pipeline():
    pipeline_steps = [
        "📥 Input",
        "🔍 Backbone",
        "🎯 FiLM",
        "👁️ Attention",
        "📊 PCVW",
        "🧩 Prototype",
        "✅ Result"
    ]
    
    cols = st.columns(len(pipeline_steps))
    for col, step in zip(cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div style="
                text-align: center; 
                padding: 0.3rem 0.1rem;
                background: rgba(255,255,255,0.7);
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
                border-radius: 8px;
                border: 1px solid rgba(226, 232, 240, 0.5);
                font-size: 0.65rem;
            ">
                <div style="font-size: 0.9rem;">{step.split()[0]}</div>
                <div style="font-size: 0.55rem; color: #718096;">{' '.join(step.split()[1:])}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>DLR-<span class="accent">PCVW</span></h1>
        <p class="subtitle">Distribution-Aware Few-Shot Learning for Acne Severity Classification</p>
        <span class="badge">🎓 Research Prototype · Thesis Demonstration</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔬 Model Information")
        st.markdown("""
        | | |
        |---|---|
        | **Framework** | DLR-PCVW |
        | **Backbone** | EfficientNet-B0 |
        | **Learning** | Few-Shot Learning |
        | **Task** | 4-Way Classification |
        | **Input** | 224 × 224 |
        | **Parameters** | 6.93M |
        | **Ensemble** | 5 Seeds |
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Shot Selection")
        k_shot = st.select_slider(
            "Select k-shot",
            options=[1, 3, 5, 10],
            value=5,
            help="Number of support examples per class"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Results (5 Seeds)")
        results_data = {
            "1-Shot": "76.65% ± 1.00%",
            "3-Shot": "79.40% ± 0.73%",
            "5-Shot": "79.60% ± 0.62%",
            "10-Shot": "80.27% ± 0.43%"
        }
        for k, v in results_data.items():
            st.markdown(f"**{k}:** {v}")
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This system is a **research prototype** and is not a clinical diagnostic tool.
        Attention visualizations indicate model focus and are not clinically validated.
        """)

    # Load Model
    @st.cache_resource
    def load_model():
        local_model = "saved_models/ensemble_models.pt"
        if os.path.exists(local_model):
            model_path = local_model
            st.info("📁 Using local model")
        else:
            model_path = "https://huggingface.co/iamthearafatkhan/dlr-pcvw-acne-severity/resolve/main/ensemble_models.pt"
            st.info("📥 Downloading model from Hugging Face...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = DLRPredictor(model_path, device)
        
        support_x, support_y = load_support_set()
        predictor.load_support_set(support_x, support_y)
        
        return predictor
    
    predictor = load_model()

    # Main content
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or select an image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image)
            
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    show_processing_pipeline()
                    
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.003)
                        progress_bar.progress(i + 1)
                    
                    result = predictor.predict(img_tensor, k_shot=k_shot)
                    attention_map = result['vis_data']['scores']
                    
                    if attention_map.ndim == 1:
                        attention_map = attention_map.reshape(7, 7)
                    elif attention_map.ndim > 2:
                        attention_map = attention_map.mean(axis=0)
                    
                    st.session_state['result'] = result
                    st.session_state['attention_map'] = attention_map
                    st.session_state['img_tensor'] = img_tensor
                    st.session_state['has_result'] = True
                    st.session_state['k_shot'] = k_shot
                    
                    st.rerun()

    with col2:
        st.markdown("### 📊 Analysis Result")
        
        if st.session_state.get('has_result', False):
            result = st.session_state['result']
            img_tensor = st.session_state['img_tensor']
            attention_map = st.session_state['attention_map']
            
            severity = result['severity_short']
            color = predictor.severity_colors[result['prediction']]
            confidence = result['confidence']
            k_shot_used = result.get('k_shot', st.session_state.get('k_shot', 5))
            
            # Get the predicted class probability
            pred_prob = result['probabilities'][result['prediction']]
            
            # Severity card
            st.markdown(f"""
            <div class="severity-card" style="border-color: {color};">
                <p style="color: #718096; font-size: 0.8rem; margin: 0;">Predicted Severity</p>
                <div class="grade" style="color: {color};">{severity}</div>
                <div class="sub">Grade {result['prediction']} · {k_shot_used}-Shot</div>
                <div class="confidence-bar">
                    <div class="fill" style="width: {confidence*100}%; background: {color};"></div>
                </div>
                <p style="margin-top: 0.3rem; font-size: 0.9rem;">Confidence: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # === NEW: Two Metric Cards - Prototype Similarity & Prediction Probability ===
            st.markdown("""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="value">{:.2%}</div>
                    <div class="label">Prototype Similarity</div>
                </div>
                <div class="metric-card">
                    <div class="value">{:.2%}</div>
                    <div class="label">Prediction Probability</div>
                </div>
            </div>
            """.format(confidence, pred_prob), unsafe_allow_html=True)
            
            # Prototype similarity bar chart
            st.markdown("#### 📈 Prototype Similarity")
            fig = plot_prototype_similarity(
                result['probabilities'],
                predictor.severity_names,
                predictor.severity_colors,
                result['prediction']
            )
            st.pyplot(fig)
            plt.close()
            
            # Attention visualization
            st.markdown("#### 🔬 Lesion Attention")
            
            try:
                img_display, overlay, attention_upsampled = create_attention_overlay(img_tensor, attention_map)
                
                fig2, axes = plt.subplots(1, 3, figsize=(9, 2.5))
                axes[0].imshow(img_display)
                axes[0].set_title('Original', fontsize=9)
                axes[0].axis('off')
                
                axes[1].imshow(attention_upsampled, cmap='hot', interpolation='bilinear')
                axes[1].set_title('Attention Map', fontsize=9)
                axes[1].axis('off')
                
                axes[2].imshow(overlay)
                axes[2].set_title('Overlay', fontsize=9)
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            except Exception as e:
                st.warning(f"Could not display attention visualization")
            
            # Explanation
            with st.expander("💡 Why this prediction?"):
                st.markdown(f"""
                - **Predicted Grade:** {severity}
                - **Closest Prototype:** {predictor.severity_names[result['prediction']]}
                - **Prototype Similarity:** {confidence:.2%}
                - **Prediction Probability:** {pred_prob:.2%}
                - **Used {k_shot_used}-Shot** support examples per class
                """)
            
            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Research Prototype:</strong> This visualization indicates regions emphasized by the model 
                and is not clinically validated lesion segmentation. Not intended for clinical diagnosis.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; color: #a0aec0; font-size: 0.75rem; padding: 0.5rem 0;">
        DLR-PCVW · Distribution-Aware Few-Shot Learning · Thesis Demonstration
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

