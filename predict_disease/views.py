# predict_disease/views.py
from django.shortcuts import render
from .utils import run_prediction_hf, run_prediction_pd
import json



# 1. Trang menu chính để chọn 1 trong 3 task
def index(request):
    return render(request, 'index.html')

# 2. Task: Dự đoán Bệnh Tim
def predict_heart(request):
    result = None
    submitted_visits = {}

    if request.method == 'POST':
        try:
            print("===== POST DATA (HEART) =====")
            print(request.POST)

            total_visits_str = request.POST.get('total_visits', '0')
            total_visits = int(total_visits_str)

            visits_data = []

            for i in range(1, total_visits + 1):

                submitted_visits[f'visit_{i}'] = request.POST.get(f'visit_{i}', '')

                raw_string = request.POST.get(f'visit_{i}', '')
                codes_list = request.POST.getlist(f'visit_{i}_codes[]')

                codes = []

                # Case 1: nhận LIST thực (JS gửi lên dạng array)
                if codes_list:
                    for v in codes_list:
                        v = v.strip()
                        if " " in v:
                            v = v.split(" ")[0]
                        if v:
                            codes.append(v)

                # Case 2: nhập text bình thường: "25000, 4019, 4280"
                elif raw_string:
                    raw_split = [c.strip() for c in raw_string.split(',') if c.strip()]
                    for v in raw_split:
                        if " " in v:
                            v = v.split(" ")[0]
                        if v:
                            codes.append(v)

                # ## ONLY APPEND if code list is not empty
                if len(codes) > 0:
                    visits_data.append(codes)

            if len(visits_data) == 0:
                raise ValueError("Không có mã bệnh hợp lệ nào được nhập.")

            prob, label, label_class = run_prediction_hf(visits_data)

            result = {
                'prediction_text': label,
                'prediction_class': label_class,
                'probability': round(prob * 100, 2),
                'num_visits': len(visits_data),
                'locked': True
            }

        except Exception as e:
            print(f"Lỗi xử lý Form Heart: {e}")
            result = {
                'prediction_text': "❗ Lỗi dữ liệu đầu vào", 
                'prediction_class': "risk-low",
                'probability': 0,
                'num_visits': 0
            }

    return render(request, 'heart.html', {
        'result': result,
        'submitted_visits_json': json.dumps(submitted_visits, ensure_ascii=False)
    })


# 3. Task: Dự đoán Parkinson
def predict_parkinson(request):
    result = None
    submitted_visits = {} # Biến lưu lại dữ liệu form

    if request.method == 'POST':
        try:
            print("===== POST DATA (PARKINSON) =====")
            
            # Logic lấy dữ liệu giống hệt predict_heart
            total_visits_str = request.POST.get('total_visits', '0')
            total_visits = int(total_visits_str)

            visits_data = []

            for i in range(1, total_visits + 1):
                # Lưu dữ liệu input
                submitted_visits[f'visit_{i}'] = request.POST.get(f'visit_{i}', '')

                raw_string = request.POST.get(f'visit_{i}', '')
                codes_list = request.POST.getlist(f'visit_{i}_codes[]')

                codes = []

                # Case 1: Dữ liệu từ Dropdown/Autocomplete
                if codes_list:
                    for v in codes_list:
                        v = v.strip()
                        if " " in v: v = v.split(" ")[0]
                        if v: codes.append(v)

                # Case 2: Dữ liệu nhập tay ngăn cách bởi dấu phẩy
                elif raw_string:
                    raw_split = [c.strip() for c in raw_string.split(',') if c.strip()]
                    for v in raw_split:
                        if " " in v: v = v.split(" ")[0]
                        if v: codes.append(v)

                if len(codes) > 0:
                    visits_data.append(codes)

            if len(visits_data) == 0:
                raise ValueError("Không có mã bệnh hợp lệ nào được nhập.")

            # --- KHÁC BIỆT DUY NHẤT Ở ĐÂY ---
            # Gọi hàm dự đoán Parkinson
            prob, label, label_class = run_prediction_pd(visits_data)
            # --------------------------------

            result = {
                'prediction_text': label,
                'prediction_class': label_class,
                'probability': round(prob * 100, 2),
                'num_visits': len(visits_data),
                'locked': True
            }

        except Exception as e:
            print(f"Lỗi xử lý Form Parkinson: {e}")
            result = {
                'prediction_text': "❗ Lỗi hoặc chưa nhập dữ liệu", 
                'prediction_class': "risk-low",
                'probability': 0,
                'num_visits': 0
            }

    return render(request, 'parkinson.html', {
        'result': result,
        'submitted_visits_json': json.dumps(submitted_visits, ensure_ascii=False)
    })

# 4. Task: Dự đoán bệnh trong lần khám tới (Next Visit)
def predict_next_visit(request):
    result = None
    if request.method == 'POST':
        # Xử lý logic AI dự đoán Next Visit ở đây
        pass
    return render(request, 'predict_disease/form_next_visit.html', {'result': result})