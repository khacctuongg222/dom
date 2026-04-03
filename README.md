<img width="1595" height="567" alt="image" src="https://github.com/user-attachments/assets/b2167902-dd2d-45e9-a46f-f288bf370dbe" /># 📋 BOM Object Detection & OCR System

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
**Detection:**

   **Backbone:** ResNet-50
   
   **Kết quả tốt nhất (sau 34 epochs):**
   - **mAP@0.5:0.95:** 0.7449
   - **mAP@0.5:** 0.9236
   - **mAR@100:** 0.8136
   - **Loss:** 0.0723
   
   **Biểu đồ quá trình huấn luyện:**
   
   ![Loss Curve](https://github.com/user-attachments/assets/29b48133-4bec-4d07-bf9f-0cf158d999af)
   
   ![Training Metrics](https://github.com/user-attachments/assets/2551d60c-376e-448b-a847-87666040f511)
   
   **Kết quả kiểm thử mô hình:**
   - Mô hình hoạt động ổn định với các vùng gán nhãn: PartDrawing, Note, Table
   
   ![Test Result 1](https://github.com/user-attachments/assets/5fe16546-01aa-476b-a0c4-92bcde49f641)
   
   ![Test Result 2](https://github.com/user-attachments/assets/f3e44823-a1cd-41f5-be0f-c638674adb6e)
   
   ---

**OCR & Xử lý ngôn ngữ:**
Hệ thống khi hoạt động riêng lẻ không qua nhiều bước thì hoạt động tốt (đưa trực tiếp bảng hoặc ảnh ghi chú vào) tuy nhiên sau khi qua pipeline hoàn chỉnh từ  Detection đến OCR ảnh bị giảm chất lượng làm cho kết quả đầu ra không tốt bằng. Đường dẫn đến Web mẫu [https://huggingface.co/spaces/khac-tuong-222/ocr]

Với thử nghiệm từ tệp OUTPUT kết quả trích xuất trực tiếp với bảng và ghi chú như sau:
Kết quả xử lý bảng: 

Ảnh đầu vào:
<img width="1485" height="828" alt="image" src="https://github.com/user-attachments/assets/41bd97e6-f98d-4bb7-85ab-7e6d54a7fc96" />
Ảnh kết quả xuất sang PDF:
<img width="1902" height="875" alt="image" src="https://github.com/user-attachments/assets/d1ee1f5f-d4dd-4225-9602-f135ed22c43e" />

Kết quả trước khi xử lý với gemini: 
<table class="table table-bordered"><thead><tr><th>14</th><th colspan="3">Bọc lột</th><th>3</th><th>Đóng manh</th><th></th></tr></thead><tbody><tr><td>13</td><td colspan="3">Vòng đềm</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>12</td><td colspan="3">Chở truy Sné=55</td><td>2</td><td>Thép CT3</td><td></td></tr><tr><td>11</td><td colspan="3">Vít M6=50</td><td>6</td><td>Thép CT3</td><td></td></tr><tr><td>10</td><td colspan="3">Bu.lông M4=20</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>9</td><td colspan="3">Vòng đềm vénnh</td><td>1</td><td>Thép 651</td><td></td></tr><tr><td>8</td><td colspan="3">Then bảng 4=4=14</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>7</td><td colspan="3">Gòng chén</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>6</td><td colspan="3">Chở chén</td><td>1</td><td>Số day</td><td></td></tr><tr><td>5</td><td colspan="3">Bọc lột</td><td>1</td><td>Đóng manh</td><td></td></tr><tr><td>4</td><td colspan="3">Giá độ</td><td>1</td><td>Gang 15-32</td><td></td></tr><tr><td>3</td><td colspan="3">Bánh răng</td><td>2</td><td>Thép 45</td><td>M=32×12</td></tr><tr><td>2</td><td colspan="3">Hệp bánh răng</td><td>1</td><td>Gang 15-32</td><td></td></tr><tr><td>1</td><td colspan="3">NĐô</td><td>1</td><td>Gang 15-32</td><td></td></tr><tr><td>Vịnh</td><td colspan="3">Tên cál hét mày</td><td>Số lg</td><td>VĐt lleu</td><td>Đài chữ</td></tr><tr><td colspan="2">Bán gốc</td><td>L.X</td><td>cū</td><td rowspan="2" colspan="3">BỒM BÁNH RẂNG</td></tr><tr><td colspan="2">Con chữ</td><td>*746</td><td>1.99</td></tr><tr><td colspan="4">Bố mn Hình hoa .VKT<br>Đgi học Bách khoa Há nghi</td><td colspan="2">Bẃn Vē LẖP SÓ 3</td><td>Tý 1ê<br>1:1</td></tr></tbody></table>
Kết quả sau khi xử lý với gemini:
<table class="table table-bordered"><thead><tr><th>14</th><th colspan="3">Bọc lót</th><th>3</th><th>Đồng thau</th><th></th></tr></thead><tbody><tr><td>13</td><td colspan="3">Vòng đệm</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>12</td><td colspan="3">Chốt truyø s6=55</td><td>2</td><td>Thép CT3</td><td></td></tr><tr><td>11</td><td colspan="3">Vít M6×50</td><td>6</td><td>Thép CT3</td><td></td></tr><tr><td>10</td><td colspan="3">Bu.lông M4×20</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>9</td><td colspan="3">Vòng đệm vênh</td><td>1</td><td>Thép 651</td><td></td></tr><tr><td>8</td><td colspan="3">Then phẳng 4×4×14</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>7</td><td colspan="3">Gối chén</td><td>1</td><td>Thép CT3</td><td></td></tr><tr><td>6</td><td colspan="3">Chốt chén</td><td>1</td><td>Sắt dây</td><td></td></tr><tr><td>5</td><td colspan="3">Bọc lót</td><td>1</td><td>Đồng thau</td><td></td></tr><tr><td>4</td><td colspan="3">Giá đỡ</td><td>1</td><td>Gang 15-32</td><td></td></tr><tr><td>3</td><td colspan="3">Bánh răng</td><td>2</td><td>Thép 45</td><td>M=3×12</td></tr><tr><td>2</td><td colspan="3">Hộp bánh răng</td><td>1</td><td>Gang 15-32</td><td></td></tr><tr><td>1</td><td colspan="3">Đồ gá</td><td>1</td><td>Gang 15-32</td><td></td></tr><tr><td>Vịnh</td><td colspan="3">Tên chi tiết máy</td><td>Số lg</td><td>Vật liệu</td><td>Đài chữ</td></tr><tr><td colspan="2">Bản gốc</td><td>L.X</td><td>cũ</td><td rowspan="2" colspan="3">BỐM BÁNH RĂNG</td></tr><tr><td colspan="2">Con chữ</td><td>*746</td><td>1.99</td></tr><tr><td colspan="4">Bố trí Hình hoa .VKT<br>Đại học Bách khoa Hà nội</td><td colspan="2">Bản vẽ LHP SÓ 3</td><td>Tỷ lệ<br>1:1</td></tr></tbody></table>
Kết quả xử lý ghi chú: 

Ảnh đầu vào:
<img width="1485" height="828" alt="image" src="https://github.com/user-attachments/assets/6a055e16-cc36-46b4-9f14-c1fb45abd027" />
Kết quả trước khi xử lý với gemini: 
      "
      Thuyết mianh : Thần bom góm các chi tiệt may chinh la
      pia do 4, hóp bánh răng 2 và náp 1, chùng duqc ghép
      khıt với nhau băng hai chôt dịnh vj 12 và sâu vít 11.
      Trong hóp bánh răng 2 có hai truc 15 và 17, trôn do
      lần chât hai bánh răng 3 : bánh răng chù dòng ở trên và
      bánh răng bị dòng ô duot ; cáp bánh răng nay quay nhô
      chuyén dòng của bánh răng bên ngoài (vê băng nét hai
      chém gách mânh, láp ô chô then ô).
      Các bánh răng 3 quay nhanh theo chiều mối tôn sê
      tao ra súc hút tù lô phila sau bom dé kéo chât läng cháy
      vao các kê răng ; lán do, chât läng chuyén theo các kê
      răng nay qua lô ra phila truve. Cú thê, chât läng duqc hút
      và dây lên túc qua bom völ ap lyc lôn.
      Các bác lột 5 và 14 là các ó trupt ô dâu hai truc -
      Các ch肘 6,7 dung dé chên khít không cho chât läng
      rô rì ra ngoài. "
Kết quả sau khi xử lý với gemini:
      "Thuyết minh: Thân bom gồm các chi tiết chính là vỏ bom 4, hộp bánh răng 2 và nắp 1, chúng được ghép khít với nhau bằng hai chốt định vị 12 và ổ vít 11.
      Trong hộp bánh răng 2 có hai trục 15 và 17, trên đó lần lượt chất hai bánh răng 3: bánh răng chủ động ở trên và bánh răng bị động ở dưới; cặp bánh răng này quay nhờ chuyển động của bánh răng bên ngoài (về        bằng nét hai chém gạch mảnh, lắp ở chỗ then ổ).
      Các bánh răng 3 quay nhanh theo chiều mũi tên sẽ tạo ra sức hút từ lỗ phía sau bom để kéo chất lỏng vào các kẽ răng; sau đó, chất lỏng chuyển theo các kẽ răng này qua lỗ ra phía trước. Cứ thế, chất lỏng          được hút và đẩy lên tức qua bom với áp lực lớn.
      Các bạc lót 5 và 14 là các ổ trục ở đầu hai trục.
      Các chốt 6, 7 dùng để chặn khít không cho chất lỏng rò rỉ ra ngoài."
      

**Toàn bộ hệ thống Detection & OCR:**

   **Kết quả quả thử nghiệm với ảnh có độ nét cao**

   Ảnh đầu vào:
   <img width="977" height="692" alt="image" src="https://github.com/user-attachments/assets/52764366-8f75-4ac0-8624-08eaf120a897" />
   Kết quả detection: 
   <img width="1595" height="567" alt="image" src="https://github.com/user-attachments/assets/3fb88c44-cbfa-4155-b1eb-12eef996f935" />
   Kết quả ocr:
   <img width="1489" height="674" alt="image" src="https://github.com/user-attachments/assets/b126c0d8-fe88-4d75-93ee-4c7af0e74330" />
   <img width="1487" height="350" alt="image" src="https://github.com/user-attachments/assets/f98b14a2-93cd-48a1-8bed-7a6de8027bb1" />
   <img width="547" height="135" alt="image" src="https://github.com/user-attachments/assets/a58769fd-65d9-48a6-a716-1a5c5f435f84" />
   <img width="569" height="172" alt="image" src="https://github.com/user-attachments/assets/33894747-91d0-4b16-b52e-13d44fde2b07" />
   **Kết quả kiểm thử nghiệm với ảnh bị nhiễu**
    Ảnh đầu vào:
   <img width="870" height="490" alt="image" src="https://github.com/user-attachments/assets/1650a32f-a5f8-44b1-b698-b15c811bb34f" />
   Kết quả detection: 
   <img width="1564" height="554" alt="image" src="https://github.com/user-attachments/assets/0ab6c2fe-56c9-437c-afb1-2d6c1bf78297" />
   Kết quả ocr:
   <img width="1509" height="840" alt="image" src="https://github.com/user-attachments/assets/12dba988-dd8f-4974-96da-057f570ba500" />
   <img width="359" height="122" alt="image" src="https://github.com/user-attachments/assets/c170a738-8db7-40c7-89c9-163875235761" />
   <img width="786" height="247" alt="image" src="https://github.com/user-attachments/assets/3b6d9056-96a5-4f3f-9bb8-614f0fe4e5d1" />
   KẾT QUẢ TRÍCH XUẤT NÀY BỊ SAI HOÀN TOÀN
   <img width="1524" height="496" alt="image" src="https://github.com/user-attachments/assets/9d901922-87a8-47f6-8217-361fba7b38de" />
 **ĐÁNH GIÁ**
 Hệ thống cần tối ưu cách truyền dữ liệu và xử lý dữ liệu đầu vào hiện tại tôi chưa tìm ra cách tối ưu tốt nhất cho toàn hệ thống 

## 🔗 Tài nguyên

| Tài nguyên | Link |
|-----------|------|
| **Model Weights** | [Google Drive](https://drive.google.com/drive/folders/1TYNrXfxgpcJpLomBqc8EfAB59wXP9fJt?usp=sharing) |
| **Web Demo full pipeline** | [Hugging Face Spaces](https://huggingface.co/spaces/khac-tuong-222/bom) |
| **Web Demo ocr** | [Hugging Face Spaces](https://huggingface.co/spaces/khac-tuong-222/ocr) |


---

## 📝 Ghi chú
- Dự án sử dụng chủ yếu  **Python**
- Khuyến nghị sử dụng GPU cho quá trình huấn luyện
- Để cải thiện tốc độ inference, hãy xem xét sử dụng GPU 
