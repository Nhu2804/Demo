import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from django.conf import settings

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = os.path.join(settings.BASE_DIR, 'predict_disease/ml_artifacts')
MODEL_HF_PATH = os.path.join(ARTIFACTS_DIR, 'model1_hf_6k.pth')
MODEL_PD_PATH = os.path.join(ARTIFACTS_DIR, 'model1_pd_6k.pth') 
MAP_PATH   = os.path.join(ARTIFACTS_DIR, 'code_map.pkl')
MAX_LEN = 40

# -------------------------------------------------------------
# MODEL DEFINITION — MUST MATCH TRAIN TIME 100%
# -------------------------------------------------------------
class RETAINPP(nn.Module):
    def __init__(self, vocab, embed_dim=128, att_dim=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(vocab, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.alpha_gru = nn.GRU(embed_dim, att_dim, batch_first=True)
        self.beta_gru  = nn.GRU(embed_dim, att_dim, batch_first=True)

        self.alpha_fc = nn.Linear(att_dim, 1)
        self.beta_fc  = nn.Linear(att_dim, vocab)

        self.output = nn.Sequential(
            nn.Linear(vocab, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, lens):
        B, T, V = x.shape

        # (B, T, V) → (B, T, embed)
        e = self.embed(x)

        # reverse time
        idx = torch.arange(T-1, -1, -1).to(x.device)
        e_rev = e[:, idx, :]

        # attention
        alpha_h,_ = self.alpha_gru(e_rev)
        beta_h,_  = self.beta_gru(e_rev)

        # scalar attention weights
        alpha = F.softmax(self.alpha_fc(alpha_h).squeeze(-1), dim=1)

        # vector code importance
        beta = torch.sigmoid(self.beta_fc(beta_h))

        # reverse back
        alpha = alpha[:, idx]
        beta  = beta[:, idx]

        # weighted feature sum
        c = torch.sum(alpha.unsqueeze(-1) * (beta * x), dim=1)

        return self.output(c).squeeze(1)

# -------------------------------------------------------------
# CACHE — tránh load lại model
# -------------------------------------------------------------
_model_hf_instance = None
_model_pd_instance = None
_code_map       = None

# -------------------------------------------------------------
# LOAD MODEL + CODE MAP
# -------------------------------------------------------------
def load_hf_resources():
    global _model_hf_instance, _code_map

    if _model_hf_instance:
        return _model_hf_instance, _code_map

    print(">>> Loading HF Model and Code Map...")

    # load code map
    if _code_map is None:
        with open(MAP_PATH, "rb") as f:
            _code_map = pickle.load(f)

    V = len(_code_map)

    model = RETAINPP(V).to(device)
    state_dict = torch.load(MODEL_HF_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _model_hf_instance = model

    print(">>> HF Model Loaded Successfully!")
    return _model_hf_instance, _code_map

# -------------------------------------------------------------
# RUN INFERENCE
# -------------------------------------------------------------
def run_prediction_hf(visits_codes):
    """
    Dự đoán Suy Tim (HF). Giữ nguyên logic cũ.
    """

    model, code_map = load_hf_resources()

    V = len(code_map)
    seq = []

    # convert visits to multi-hot
    for codes in visits_codes:
        vector = np.zeros(V, dtype=np.float32)
        for c in codes:
            if not c:
                continue
            c = c.strip()
            if " " in c:
                c = c.split(" ")[0]
            if c in code_map:
                vector[code_map[c]] = 1.0
        seq.append(vector)

    L = len(seq)

    # padding / truncating
    if L > MAX_LEN:
        seq = seq[-MAX_LEN:]
        L = MAX_LEN
    else:
        pad = [np.zeros(V)]*(MAX_LEN-L)
        seq = pad + seq

    # make tensor
    x = torch.tensor(np.stack(seq)[None,:], dtype=torch.float32).to(device)
    lens = torch.tensor([L], dtype=torch.long).to(device)

    # forward
    with torch.no_grad():
        prob = torch.sigmoid(model(x, lens)).item()

    # map prob to human label
    if prob >= 0.8:
        label_text = "NGUY CƠ RẤT CAO"
        label_class = "risk-high"
    elif prob >= 0.5:
        label_text = "CÓ NGUY CƠ"
        label_class = "risk-medium"
    else:
        label_text = "NGUY CƠ THẤP"
        label_class = "risk-low"

    return prob, label_text, label_class


# =============================================================
# PHẦN THÊM MỚI: CHO PARKINSON (PD)
# =============================================================

def load_pd_resources():
    global _model_pd_instance, _code_map

    # Nếu đã load PD model rồi thì trả về luôn
    if _model_pd_instance:
        return _model_pd_instance, _code_map

    print(">>> Loading PD Model...")

    if _code_map is None:
        with open(MAP_PATH, "rb") as f:
            _code_map = pickle.load(f)

    V = len(_code_map)

    model = RETAINPP(V).to(device)
    # Load model PD riêng
    if os.path.exists(MODEL_PD_PATH):
        state_dict = torch.load(MODEL_PD_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        _model_pd_instance = model
        print(">>> PD Model Loaded Successfully!")
    else:
        print(f"!!! Error: PD Model not found at {MODEL_PD_PATH}")
        return None, _code_map # Tránh crash nếu thiếu file

    return _model_pd_instance, _code_map

def run_prediction_pd(visits_codes):
    """
    Dự đoán Parkinson. 
    Logic xử lý dữ liệu giống hệt run_prediction nhưng dùng model PD.
    """
    model, code_map = load_pd_resources()

    if model is None:
        return 0.0, "LỖI MODEL", "error"

    V = len(code_map)
    seq = []

    # convert visits to multi-hot
    for codes in visits_codes:
        vector = np.zeros(V, dtype=np.float32)
        for c in codes:
            if not c: continue
            c = c.strip()
            if " " in c: c = c.split(" ")[0]
            if c in code_map:
                vector[code_map[c]] = 1.0
        seq.append(vector)

    L = len(seq)

    # padding / truncating
    if L > MAX_LEN:
        seq = seq[-MAX_LEN:]
        L = MAX_LEN
    else:
        pad = [np.zeros(V)]*(MAX_LEN-L)
        seq = pad + seq

    # make tensor
    x = torch.tensor(np.stack(seq)[None,:], dtype=torch.float32).to(device)
    lens = torch.tensor([L], dtype=torch.long).to(device)

    # forward
    with torch.no_grad():
        prob = torch.sigmoid(model(x, lens)).item()

    # map prob to human label (Dùng ngưỡng khác cho Parkinson)
    # Ví dụ: Ngưỡng 0.5, bạn nên chỉnh lại theo kết quả train
    if prob >= 0.8:
        label_text = "NGUY CƠ RẤT CAO"
        label_class = "risk-high"
    elif prob >= 0.5: 
        label_text = "CÓ NGUY CƠ"
        label_class = "risk-medium"
    else:
        label_text = "NGUY CƠ THẤP"
        label_class = "risk-low"

    return prob, label_text, label_class