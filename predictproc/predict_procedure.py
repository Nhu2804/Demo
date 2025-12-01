import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from django.conf import settings

# =========================================================
# 1. ĐỊNH NGHĨA MODEL (PHẢI GIỐNG Y HỆT FILE TRAIN)
# =========================================================
class MiMELayerUp(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gate = nn.Linear(in_dim, hidden_dim)
        self.inter = nn.Linear(in_dim, hidden_dim)

        self.bn_proj  = nn.BatchNorm1d(hidden_dim)
        self.bn_gate  = nn.BatchNorm1d(hidden_dim)
        self.bn_inter = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        if in_dim != hidden_dim:
            self.res_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.res_proj = None

        self.act = nn.GELU()

    def forward(self, x):
        h = self.proj(x)
        g = self.gate(x)
        i = self.inter(x)

        # BatchNorm yêu cầu input (B, C) - Batch size > 1
        # Trong inference 1 sample, ta phải xử lý kỹ hoặc dùng model.eval()
        h = self.bn_proj(h)
        g = self.bn_gate(g)
        i = self.bn_inter(i)

        h = self.act(h)
        g = torch.sigmoid(g)
        i = self.act(i)

        out = h * g + i
        out = self.dropout(out)

        if self.res_proj is not None:
            res = self.res_proj(x)
        else:
            res = x
        return out + res

class MiMEUpgraded(nn.Module):
    def __init__(self, dx_dim=2565, hidden=256, proc_dim=870, dropout=0.2):
        super().__init__()
        self.layer1 = MiMELayerUp(dx_dim,   hidden, dropout=dropout)
        self.layer2 = MiMELayerUp(hidden,   hidden, dropout=dropout)
        self.layer3 = MiMELayerUp(hidden,   hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, proc_dim)
        )

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        logits = self.head(h3)
        return logits

# =========================================================
# 2. KHỞI TẠO & LOAD RESOURCES
# =========================================================
MODEL = None
CODE2ID_ICD = {}   # Map: "250.00" -> 5
ID2CODE_PROC = {}  # Map: 10 -> "38.93"
PROC_NAMES = {}    # (Tùy chọn) Map code thủ thuật sang tên tiếng Việt/Anh nếu có file riêng

DEVICE = torch.device("cpu") # Web chạy CPU cho nhẹ
TRAIN_DX_DIM = 2565
TRAIN_PROC_DIM = 870

def load_resources():
    global MODEL, CODE2ID_ICD, ID2CODE_PROC, PROC_NAMES
    
    # Đường dẫn đến thư mục chứa model và pickle
    base_path = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(base_path, 'ml_resources')
    
    print(f"📂 Loading ML resources from: {resource_dir}")

    try:
        # --- A. LOAD MODEL ---
        MODEL = MiMEUpgraded(dx_dim=TRAIN_DX_DIM, hidden=256, proc_dim=TRAIN_PROC_DIM)
        weight_path = os.path.join(resource_dir, 'mime_finetuned.pth')
        
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=DEVICE)
            MODEL.load_state_dict(state_dict)
            MODEL.to(DEVICE)
            MODEL.eval() # <--- QUAN TRỌNG: Tắt Dropout và khóa BatchNorm
            print("✅ Model loaded successfully.")
        else:
            print(f"❌ Model file missing: {weight_path}")

        # --- B. LOAD ICD MAP (Pickle) ---
        icd_map_path = os.path.join(resource_dir, 'diagnosis_map.pkl')
        if os.path.exists(icd_map_path):
            with open(icd_map_path, "rb") as f:
                CODE2ID_ICD = pickle.load(f)
            print(f"✅ Loaded ICD Map: {len(CODE2ID_ICD)} codes.")
        else:
            print(f"❌ Missing diagnosis_map.pkl")

        # --- C. LOAD PROC MAP (Pickle) ---
        proc_map_path = os.path.join(resource_dir, 'procedure_map.pkl')
        if os.path.exists(proc_map_path):
            with open(proc_map_path, "rb") as f:
                code2id_proc = pickle.load(f)
                # Đảo chiều để tra ngược từ ID ra Code (Index 5 -> "38.93")
                ID2CODE_PROC = {v: k for k, v in code2id_proc.items()}
            print(f"✅ Loaded Proc Map: {len(ID2CODE_PROC)} codes.")
        else:
            print(f"❌ Missing procedure_map.pkl")
            
        # --- D. LOAD TÊN THỦ THUẬT (Để hiển thị trên UI cho đẹp) ---
        # Nếu bạn có file JSON chứa tên đầy đủ (VD: "38.93" -> "Thông tim")
        # Thì load vào đây. Nếu không thì dùng code làm tên luôn.
        proc_list_path = os.path.join(resource_dir, 'procedure_list.json')
        if os.path.exists(proc_list_path):
             import json
             with open(proc_list_path, 'r', encoding='utf-8') as f:
                 raw_list = json.load(f)
                 # Chuyển list thành dict để tra cứu cho nhanh
                 for item in raw_list:
                     if isinstance(item, dict):
                        PROC_NAMES[str(item['code'])] = item.get('name', str(item['code']))

    except Exception as e:
        print(f"❌ ERROR loading resources: {e}")

# Load ngay khi chạy server
load_resources()

# =========================================================
# 3. HÀM DỰ ĐOÁN (INFERENCE)
# =========================================================
def predict_procedure_from_diag(diag_codes_list):
    """
    Input: Danh sách mã ICD (list of strings) ['250.00', '401.9']
    Output: Danh sách thủ thuật (Số lượng động dựa trên độ tin cậy)
    """
    if MODEL is None:
        return []

    # 1. Tạo vector input
    x_input = np.zeros(TRAIN_DX_DIM, dtype=np.float32)
    valid_input = False
    
    for code in diag_codes_list:
        clean_code = str(code).strip()
        if clean_code in CODE2ID_ICD:
            idx = CODE2ID_ICD[clean_code]
            if idx < TRAIN_DX_DIM:
                x_input[idx] = 1.0
                valid_input = True
    
    if not valid_input:
        return [] # Không tìm thấy mã bệnh nào hợp lệ

    # 2. Chạy Model
    x_tensor = torch.tensor(x_input).unsqueeze(0).to(DEVICE)
    MODEL.eval()
    
    with torch.no_grad():
        logits = MODEL(x_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # ====================================================
    # 3. LỌC THÔNG MINH (SMART FILTERING)
    # ====================================================
    
    # Sắp xếp tất cả các mã theo điểm từ cao xuống thấp
    sorted_indices = probs.argsort()[::-1]
    
    results = []
    
    # CẤU HÌNH NGƯỠNG
    # Trong code train bạn tìm được BEST_THRESHOLD = 0.21 (tức 21%)
    # Nhưng khi chạy thực tế, nên để thấp hơn chút để gợi ý rộng hơn (VD: 0.15)
    THRESHOLD = 0.15 
    MAX_ITEMS = 15   # Tối đa chỉ lấy 15 (đề phòng lỗi ra quá nhiều)
    MIN_ITEMS = 2    # Tối thiểu lấy 2 (để bác sĩ có cái tham khảo, trừ khi điểm quá thấp)

    for idx in sorted_indices:
        score = float(probs[idx])
        
        # ĐIỀU KIỆN DỪNG 1: Điểm quá thấp (dưới 1%) -> Dừng ngay lập tức
        # Dù chưa đủ số lượng tối thiểu cũng bỏ, vì rác quá.
        if score < 0.01: 
            break
            
        # LOGIC QUYẾT ĐỊNH LẤY:
        # Lấy nếu: (Điểm cao hơn Ngưỡng) HOẶC (Chưa đủ số lượng tối thiểu)
        should_take = False
        
        if score >= THRESHOLD:
            should_take = True
        elif len(results) < MIN_ITEMS:
            # Nếu chưa đủ 2 kết quả, ta chấp nhận lấy thêm các kết quả thấp hơn (nhưng phải > 1%)
            # Để tránh màn hình bị trắng trơn
            should_take = True
            
        if should_take:
            if idx in ID2CODE_PROC:
                p_code = ID2CODE_PROC[idx]
                p_name = PROC_NAMES.get(p_code, f"Thủ thuật {p_code}")
                
                results.append({
                    "code": p_code,
                    "name": p_name,
                    "score": round(score * 100, 2)
                })
        else:
            # Nếu điểm đã thấp hơn ngưỡng VÀ đã đủ số lượng tối thiểu -> Dừng vòng lặp
            break
            
        # ĐIỀU KIỆN DỪNG 2: Đã lấy quá nhiều (VD: > 15 mã) -> Dừng
        if len(results) >= MAX_ITEMS:
            break

    return results