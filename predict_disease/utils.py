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
MODEL_HF_PATH = os.path.join(ARTIFACTS_DIR, 'model1_hf.pth')
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
        label_text = "VERY HIGH RISK"
        label_class = "risk-high"
    elif prob >= 0.5:
        label_text = "AT RISK"
        label_class = "risk-medium"
    else:
        label_text = "LOW RISK"
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
        label_text = "VERY HIGH RISK"
        label_class = "risk-high"
    elif prob >= 0.5: 
        label_text = "AT RISK"
        label_class = "risk-medium"
    else:
        label_text = "LOW RISK"
        label_class = "risk-low"

    return prob, label_text, label_class


# # =============================================================
# # =============================================================
# # PHẦN THÊM MỚI: DỰ ĐOÁN LẦN KHÁM TIẾP THEO (NEXT VISIT - BERT)
# # =============================================================
# # =============================================================

# import pytorch_pretrained_bert as Bert

# # Đường dẫn model BERT
# MODEL_BERT_PATH = os.path.join(ARTIFACTS_DIR, 'model_behrt_next_visit.pth')

# # --- 1. CÁC CLASS CẤU TRÚC MODEL BERT ---
# # (Giữ nguyên cấu trúc Class, chỉ thay đổi cách gọi ở phần Load)

# class BertEmbeddings(nn.Module):
#     """Construct the embeddings from word, segment, age"""
#     def __init__(self, config, feature_dict=None):
#         super(BertEmbeddings, self).__init__()

#         if feature_dict is None:
#             self.feature_dict = {
#                 'word': True, 'seg': True, 'age': True, 'position': True
#             }
#         else:
#             self.feature_dict = feature_dict

#         if feature_dict['word']:
#             self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

#         if feature_dict['seg']:
#             self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)

#         if feature_dict.get('age'): # Fix: dùng .get để tránh lỗi nếu key không tồn tại
#             self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)

#         if feature_dict['position']:
#             self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
#             from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

#         self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None):
#         embeddings = self.word_embeddings(word_ids)

#         if self.feature_dict['seg'] and seg_ids is not None:
#             segment_embed = self.segment_embeddings(seg_ids)
#             embeddings = embeddings + segment_embed

#         if self.feature_dict.get('age') and age_ids is not None:
#             age_embed = self.age_embeddings(age_ids)
#             embeddings = embeddings + age_embed

#         if self.feature_dict['position'] and posi_ids is not None:
#             posi_embeddings = self.posi_embeddings(posi_ids)
#             embeddings = embeddings + posi_embeddings

#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

#     def _init_posi_embedding(self, max_position_embedding, hidden_size):
#         def even_code(pos, idx):
#             return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

#         def odd_code(pos, idx):
#             return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

#         lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)
#         for pos in range(max_position_embedding):
#             for idx in np.arange(0, hidden_size, step=2):
#                 lookup_table[pos, idx] = even_code(pos, idx)
#         for pos in range(max_position_embedding):
#             for idx in np.arange(1, hidden_size, step=2):
#                 lookup_table[pos, idx] = odd_code(pos, idx)
#         return torch.tensor(lookup_table)


# class BertModel(Bert.modeling.BertPreTrainedModel):
#     def __init__(self, config, feature_dict):
#         super(BertModel, self).__init__(config)
#         self.embeddings = BertEmbeddings(config=config, feature_dict=feature_dict)
#         self.encoder = Bert.modeling.BertEncoder(config=config)
#         self.pooler = Bert.modeling.BertPooler(config)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None,
#                 output_all_encoded_layers=True):
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
#         encoded_layers = self.encoder(embedding_output,
#                                       extended_attention_mask,
#                                       output_all_encoded_layers=output_all_encoded_layers)
#         sequence_output = encoded_layers[-1]
#         pooled_output = self.pooler(sequence_output)
#         if not output_all_encoded_layers:
#             encoded_layers = encoded_layers[-1]
#         return encoded_layers, pooled_output


# class BertForMultiLabelPrediction(Bert.modeling.BertPreTrainedModel):
#     def __init__(self, config, num_labels, feature_dict):
#         super(BertForMultiLabelPrediction, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config, feature_dict)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
#                                      output_all_encoded_layers=False)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         if labels is not None:
#             loss_fct = nn.MultiLabelSoftMarginLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
#             return loss, logits
#         else:
#             return logits


# class BertConfig(Bert.modeling.BertConfig):
#     def __init__(self, config):
#         super(BertConfig, self).__init__(
#             vocab_size_or_config_json_file=config.get('vocab_size'),
#             hidden_size=config['hidden_size'],
#             num_hidden_layers=config.get('num_hidden_layers'),
#             num_attention_heads=config.get('num_attention_heads'),
#             intermediate_size=config.get('intermediate_size'),
#             hidden_act=config.get('hidden_act'),
#             hidden_dropout_prob=config.get('hidden_dropout_prob'),
#             attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
#             max_position_embeddings = config.get('max_position_embedding'),
#             initializer_range=config.get('initializer_range'),
#         )
#         self.seg_vocab_size = config.get('seg_vocab_size')
#         self.age_vocab_size = config.get('age_vocab_size')


# # --- 2. HÀM LOAD MODEL VÀ DỰ ĐOÁN ---

# _model_bert_instance = None # Cache model BERT

# def load_bert_resources():
#     global _model_bert_instance, _code_map

#     if _model_bert_instance:
#         return _model_bert_instance, _code_map

#     print(">>> Loading BERT Next Visit Model...")
    
#     if _code_map is None:
#         with open(MAP_PATH, "rb") as f:
#             _code_map = pickle.load(f)
    
#     # --- CẤU HÌNH ĐƯỢC CẬP NHẬT THEO FILE LOG LỖI ---
#     bert_params = {
#         'vocab_size': 2871,         # LOG BÁO: checkpoint shape là 2871
#         'hidden_size': 288,         # LOG BÁO: checkpoint shape là 288
#         'num_hidden_layers': 4,     # Giữ nguyên (nếu lỗi layer mismatch thì chỉnh tiếp)
#         'num_attention_heads': 4,   # Giữ nguyên
#         'intermediate_size': 512,   # LOG BÁO: 512 (khớp)
#         'hidden_act': 'gelu',
#         'hidden_dropout_prob': 0.1,
#         'attention_probs_dropout_prob': 0.1,
#         'max_position_embedding': 1500, # LOG BÁO: checkpoint shape là 1500
#         'initializer_range': 0.02,
#         'seg_vocab_size': 34,           # LOG BÁO: checkpoint shape là 34
#         'age_vocab_size': 120,
#     }
    
#     config = BertConfig(bert_params)
    
#     # LOG BÁO: Missing key 'age_embeddings' -> Tắt age đi
#     feature_dict = {
#         'word': True,
#         'seg': True,
#         'age': False,      # <--- QUAN TRỌNG: Tắt tính năng tuổi để khớp model
#         'position': True
#     }
    
#     model = BertForMultiLabelPrediction(config, num_labels=bert_params['vocab_size'], feature_dict=feature_dict)
    
#     if os.path.exists(MODEL_BERT_PATH):
#         try:
#             state_dict = torch.load(MODEL_BERT_PATH, map_location=device)
#             if isinstance(state_dict, nn.Module):
#                  state_dict = state_dict.state_dict()
            
#             # Load strict=True để đảm bảo khớp 100%
#             model.load_state_dict(state_dict, strict=True) 
#             model.to(device)
#             model.eval()
#             _model_bert_instance = model
#             print(">>> BERT Model Loaded Successfully!")
#         except Exception as e:
#             print(f"!!! Error loading BERT weights: {e}")
#             return None, _code_map
#     else:
#         print(f"!!! Error: BERT Model file not found at {MODEL_BERT_PATH}")
#         return None, _code_map

#     return _model_bert_instance, _code_map

# def run_prediction_next_visit(visits_codes, top_k=10):
#     model, code_map = load_bert_resources()
    
#     if model is None:
#         return []

#     id2code = {v: k for k, v in code_map.items()}

#     input_ids = []
#     seg_ids = []
#     posi_ids = []
#     # Age ids không cần nữa vì model đã tắt feature này
    
#     current_pos = 0

#     for codes in visits_codes:
#         for code in codes:
#             if not code: continue
#             code = code.strip().split(" ")[0]
            
#             if code in code_map:
#                 idx = code_map[code]
                
#                 # Check bounds vocab size để tránh lỗi index out of range nếu code_map lớn hơn model
#                 if idx < model.config.vocab_size:
#                     input_ids.append(idx)
#                     seg_ids.append(1)         
#                     posi_ids.append(current_pos)
#                     current_pos += 1

#     if not input_ids:
#         return []

#     # Cắt hoặc Pad
#     max_len = model.config.max_position_embeddings # Bây giờ là 1500
    
#     if len(input_ids) > max_len:
#         input_ids = input_ids[-max_len:]
#         seg_ids   = seg_ids[-max_len:]
#         posi_ids  = list(range(len(input_ids))) # Reset pos
#     else:
#         posi_ids = list(range(len(input_ids)))

#     input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
#     seg_ids_t   = torch.tensor([seg_ids], dtype=torch.long).to(device)
#     posi_ids_t  = torch.tensor([posi_ids], dtype=torch.long).to(device)
    
#     # Age IDs pass None hoặc dummy (vì feature_dict['age'] = False nên model sẽ bỏ qua)
#     age_ids_t = None 

#     attention_mask = torch.ones_like(input_ids_t).to(device)

#     with torch.no_grad():
#         logits = model(input_ids_t, age_ids_t, seg_ids_t, posi_ids_t, attention_mask)
#         probs = torch.sigmoid(logits)
#         top_probs, top_indices = torch.topk(probs, top_k)

#     results = []
#     top_probs_np = top_probs.cpu().numpy()[0]
#     top_indices_np = top_indices.cpu().numpy()[0]

#     for prob, idx in zip(top_probs_np, top_indices_np):
#         code_str = id2code.get(idx, f"UNK_{idx}")
#         if code_str in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
#             continue

#         results.append({
#             'code': code_str,
#             'probability': round(float(prob) * 100, 2)
#         })

#     return results