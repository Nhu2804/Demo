from django.shortcuts import render
from django.http import JsonResponse
from .predict_procedure import predict_procedure_from_diag

def procedure_page(request):
    """Render trang HTML giao diện"""
    return render(request, "procedure_page.html")

def procedure_api(request):
    """API nhận request AJAX từ frontend"""
    if request.method == "POST":
        raw_data = request.POST.get("icd_codes", "")
        
        # Tách chuỗi bằng dấu phẩy, xóa khoảng trắng thừa
        # Ví dụ: "41071, 0389 " -> ['41071', '0389']
        diag_codes = [c.strip() for c in raw_data.split(",") if c.strip()]
        
        if not diag_codes:
            return JsonResponse({"results": []})

        # Gọi hàm dự đoán
        try:
            results = predict_procedure_from_diag(diag_codes)
            return JsonResponse({"results": results}, safe=False)
        except Exception as e:
            print(f"Error in API: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)