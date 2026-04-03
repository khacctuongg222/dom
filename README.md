# 📋 Hệ Thống Phát Hiện BOM và Trích Xuất OCR

Một hệ thống toàn diện để phát hiện Bill of Materials (BOM), trích xuất văn bản (OCR) và sửa lỗi chính tả sử dụng Faster R-CNN, GLM, và Google Gemini API.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/98.8%25-Jupyter%20Notebook-orange.svg)]()
[![Python](https://img.shields.io/badge/1.2%25-Python-blue.svg)]()

---

## 📖 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Các Tính Năng](#các-tính-năng)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Kết Quả](#kết-quả)
- [Demo & Tài Nguyên](#demo--tài-nguyên)
- [Đánh Giá](#đánh-giá)
- [Ghi Chú](#ghi-chú)

---

## 🎯 Giới Thiệu

Dự án này phát triển một hệ thống AI toàn diện cho xử lý bản vẽ kỹ thuật:

| Thành Phần | Mô Tả |
|-----------|-------|
| **Phát Hiện (Detection)** | Sử dụng Faster R-CNN ResNet-50 để phát hiện các thành phần trong bản vẽ (PartDrawing, Note, Table) |
| **Trích Xuất OCR** | Trích xuất văn bản từ các vùng phát hiện được |
| **Sửa Lỗi** | Sử dụng Gemini/GLM để chuẩn hóa chính tả và thuật ngữ kỹ thuật |

---

## ✨ Các Tính Năng

- ✅ Phát hiện tự động các thành phần trong bản vẽ kỹ thuật
- ✅ Trích xuất OCR chính xác cao từ ảnh
- ✅ Sửa lỗi chính tả tự động và chuẩn hóa thuật ngữ
- ✅ Giao diện web thân thiện (Gradio)
- ✅ Hỗ trợ GPU để tăng tốc độ xử lý
- ✅ Demo full pipeline + OCR riêng biệt

---

## 🛠 Cài Đặt

### Yêu Cầu Hệ Thống

- **Python:** 3.10 hoặc cao hơn
- **OS:** Linux/Docker (khuyến nghị)
- **GPU:** NVIDIA CUDA (tuỳ chọn nhưng khuyến nghị)

### Bước 1: Clone Repository

```bash
git clone https://github.com/khacctuongg222/dom.git
cd dom
```

### Bước 2: Cài Đặt Thư Viện Hệ Thống (Linux/Docker)

```bash
sudo apt-get update && sudo apt-get install -y libgl1 python3-dev build-essential
```

### Bước 3: Cài Đặt Thư Viện Python

```bash
pip install -r requirements.txt
```

**Thư viện chính bao gồm:**
- `torch`, `torchvision` — Deep Learning Framework
- `gradio` — Web UI
- `google-genai` — Gemini API
- `transformers` — Model transformers
- `opencv-python-headless` — Computer Vision

---

## 🚀 Hướng Dẫn Sử Dụng

### 1️⃣ Huấn Luyện Model

#### Thông Tin Model
- **Kiến Trúc:** Faster R-CNN với backbone ResNet-50 FPN
- **Dataset Format:** COCO (`instances_default.json`)
- **Notebook:** `faster_rcnn_colab.ipynb`

#### Quy Trình

1. Mở file `.ipynb` trên **Google Colab**
2. Kết nối **Google Drive** để tải dataset (`data.zip`)
3. Chạy các cell để bắt đầu huấn luyện (mặc định: 50 epochs)
4. Lưu model weights: `/content/best_faster_rcnn_bom.pth`

### 2️⃣ Chạy Inference (Web Demo)

#### Chuẩn Bị

1. Tải model weights từ [Google Drive](https://drive.google.com/drive/folders/1TYNrXfxgpcJpLomBqc8EfAB59wXP9fJt?usp=sharing)
2. Đặt file `faster_rcnn_resnet50_bom.pth` vào cùng thư mục với `app.py`

> **Lưu Ý:** File được lưu là `best_faster_rcnn_bom.pth`, cần đổi lại thành `faster_rcnn_resnet50_bom.pth`

#### Thiết Lập API Key Gemini

**Cách 1 - Environment Variable (Khuyến Nghị)**
```bash
export GEMINI_API_KEY="your_api_key_here"
python app.py
```

**Cách 2 - Ghi Trực Tiếp trong Code**
```python
# Trong file app.py
GEMINI_API_KEY = "your_api_key_here"
```

#### Chạy Ứng Dụng

```bash
python app.py
```

Truy cập: `http://localhost:7860`

#### Tính Năng Giao Diện

- 📤 Tải ảnh lên để phát hiện
- 🎚 Điều chỉnh Confidence Threshold
- 🔍 Xem kết quả Detection
- 📊 Xuất kết quả JSON
- ✏️ Sửa lỗi chính tả tự động

---

## 📊 Kết Quả

### Detection Performance (Faster R-CNN)

| Metric | Giá Trị |
|--------|--------|
| **mAP@0.5:0.95** | 0.7449 |
| **mAP@0.5** | 0.9236 |
| **mAR@100** | 0.8136 |
| **Loss** | 0.0723 |
| **Epochs** | 34 |
| **Backbone** | ResNet-50 |

### Biểu Đồ Huấn Luyện

![Loss Curve](https://github.com/user-attachments/assets/29b48133-4bec-4d07-bf9f-0cf158d999af)

![Training Metrics](https://github.com/user-attachments/assets/2551d60c-376e-448b-a847-87666040f511)

### Kết Quả Kiểm Thử

![Test Result 1](https://github.com/user-attachments/assets/5fe16546-01aa-476b-a0c4-92bcde49f641)

![Test Result 2](https://github.com/user-attachments/assets/f3e44823-a1cd-41f5-be0f-c638674adb6e)

---

## 🌐 Demo & Tài Nguyên

| Tài Nguyên | Link |
|-----------|------|
| **Model Weights** | [Google Drive](https://drive.google.com/drive/folders/1TYNrXfxgpcJpLomBqc8EfAB59wXP9fJt?usp=sharing) |
| **Demo Full Pipeline** | [Hugging Face Spaces](https://huggingface.co/spaces/khac-tuong-222/bom) |
| **Demo OCR Riêng** | [Hugging Face Spaces](https://huggingface.co/spaces/khac-tuong-222/ocr) |

---

## 📈 Đánh Giá & Nhận Xét

### Điểm Mạnh
- ✅ Model Detection hoạt động ổn định, độ chính xác cao
- ✅ Các lớp gán nhãn rõ ràng: PartDrawing, Note, Table
- ✅ OCR riêng lẻ cho kết quả tốt

### Thách Thức
⚠️ **Pipeline Toàn Diện:**
- Chất lượng ảnh giảm khi qua nhiều bước xử lý
- Kết quả đầu ra không tối ưu bằng khi xử lý riêng lẻ
- Cần tối ưu hóa cách truyền dữ liệu giữa các stage

### Kế Hoạch Cải Thiện
- 🔧 Tối ưu pipeline xử lý dữ liệu
- 🎯 Cải thiện chất lượng ảnh đầu vào
- 🚀 Tùng chỉnh tham số model cho pipeline

---

## 📝 Ghi Chú

- 💾 Dự án sử dụng chủ yếu **Jupyter Notebook (98.8%)** và **Python (1.2%)**
- 🎮 **GPU khuyến nghị** cho quá trình huấn luyện (giảm thời gian từ giờ xuống phút)
- ⚡ Chạy inference trên **CPU** vẫn có thể, tốc độ sẽ chậm hơn
- 📚 Tham khảo thêm về [Faster R-CNN](https://arxiv.org/abs/1506.01497)

---

## 📄 License

MIT License - Xem [LICENSE](LICENSE) để biết chi tiết

---

## 👨‍💻 Tác Giả

**Nguyễn Khắc Tưởng** - [@khacctuongg222](https://github.com/khacctuongg222)

---

**Cập nhật lần cuối:** 03/04/2026