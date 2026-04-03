# 📋 BOM Object Detection & OCR System

Hệ thống phát hiện BOM (Bill of Materials) và trích xuất OCR sử dụng mô hình Faster R-CNN ResNet50, GLM, kết hợp API Google Gemini để sửa lỗi chính tả và thuật ngữ.

---

## 📖 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Cài đặt môi trường](#-1-cài-đặt-môi-trường)
- [Huấn luyện Model](#-2-huấn-luyện-model-train)
- [Chạy Inference](#-3-chạy-inference-pipeline-web-demo)
- [Kết quả & Báo cáo](#-báo-cáo-ngắn)
- [Tài nguyên](#-tài-nguyên)

---

## 🎯 Giới thiệu

Dự án này phát triển một hệ thống toàn diện để:
- **Phát hiện** các thành phần trong bản vẽ kỹ thuật (Part Drawing, Note, Table) sử dụng **Faster R-CNN**
- **Trích xuất OCR** từ các vùng phát hiện được
- **Sửa lỗi** chính tả và chuẩn hóa thuật ngữ sử dụng **Gemini/GLM**

---

## 🛠 1. Cài đặt môi trường

### Yêu cầu
- Python 3.10+
- Linux/Docker (khuyên dùng)

### Bước cài đặt

**1. Cài đặt các thư viện hệ thống (Linux/Docker):**
```bash
sudo apt-get update && sudo apt-get install -y libgl1 python3-dev build-essential
```

**2. Cài đặt thư viện Python:**
```bash
pip install -r requirements.txt
```

**Các thư viện chính:**
- `torch`, `torchvision` - Deep Learning Framework
- `gradio` - Web UI
- `google-genai` - Gemini API
- `transformers` - Model transformers
- `opencv-python-headless` - Computer Vision

---

## 🚀 2. Huấn luyện Model (Train)

### Thông tin chung
- **Kiến trúc:** Faster R-CNN với backbone ResNet-50 FPN
- **Dataset:** COCO format (`instances_default.json`)
- **File notebook:** `faster_rcnn_colab.ipynb`

### Quy trình huấn luyện

1. Mở file `.ipynb` trên **Google Colab**
2. Kết nối **Google Drive** để tải dataset (`data.zip`)
3. Chạy các cell để bắt đầu huấn luyện (mặc định: 50 epochs)
4. Kết quả: Model weights được lưu tại `/content/best_faster_rcnn_bom.pth`

---

## 🔍 3. Chạy Inference Pipeline (Web Demo)

### Chuẩn bị
1. Đảm bảo file weight `faster_rcnn_resnet50_bom.pth` nằm cùng thư mục với `app.py`
   > **Lưu ý:** Tên mô hình sau khi huấn luyện là `best_faster_rcnn_bom.pth` sau đó được đổi lại thành `faster_rcnn_resnet50_bom.pth`

### Thiết lập API Key Gemini

**Cách 1 - Sử dụng environment variable:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Cách 2 - Ghi trực tiếp trong code:**
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
```
Sửa trong file `app.py`

### Chạy ứng dụng
```bash
python app.py
```

### Tính năng
- Tải ảnh lên giao diện
- Điều chỉnh ngưỡng tin cậy (Confidence)
- Xem kết quả detection và mã JSON chi tiết

---

## 📊 Báo cáo ngắn

### 1. Cách tiếp cận

**Detection:**
- Sử dụng Faster R-CNN vì khả năng chính xác cao trong việc xác định các vùng đối tượng nhỏ và phức tạp trong bản vẽ kỹ thuật

**OCR & Xử lý ngôn ngữ:**
- Pipeline kết hợp phát hiện vùng chứa văn bản
- Sử dụng các mô hình ngôn ngữ lớn (Gemini/GLM) để trích xuất thông tin ngữ cảnh
- Web demo chạy trên CPU nên tốc độ được tối ưu hơn

### 2. Kết quả thử nghiệm

**Backbone:** ResNet-50

**Kết quả tốt nhất (sau 34 epochs):**
- **mAP@0.5:0.95:** 0.7449
- **mAP@0.5:** 0.9236
- **mAR@100:** 0.8136
- **Loss:** 0.0723

**Biểu đồ quá trình huấn luyện:**

![Loss Curve](https://github.com/user-attachments/assets/29b48133-4bec-4d07-bf9f-0cf158d999af)

![Training Metrics](https://github.com/user-attachments/assets/2551d60c-376e-448b-a847-87666040f511)

**Kết quả kiểm thử:**
- Mô hình hoạt động ổn định với các vùng gán nhãn: PartDrawing, Note, Table

![Test Result 1](https://github.com/user-attachments/assets/5fe16546-01aa-476b-a0c4-92bcde49f641)

![Test Result 2](https://github.com/user-attachments/assets/f3e44823-a1cd-41f5-be0f-c638674adb6e)

---

## 🔗 Tài nguyên

| Tài nguyên | Link |
|-----------|------|
| **Model Weights** | [Google Drive](https://drive.google.com/drive/folders/1TYNrXfxgpcJpLomBqc8EfAB59wXP9fJt?usp=sharing) |
| **Web Demo** | [Hugging Face Spaces](https://huggingface.co/spaces/khac-tuong-222/bom) |

---

## 📝 Ghi chú
- Dự án sử dụng chủ yếu **Jupyter Notebook** (98.8%) và **Python** (1.2%)
- Khuyến nghị sử dụng GPU cho quá trình huấn luyện
- Để cải thiện tốc độ inference, hãy xem xét sử dụng GPU hoặc quantization model