Hướng dẫn dự án: BOM Object Detection & OCR
Dự án này sử dụng mô hình Faster R-CNN để phát hiện các thành phần trong bản vẽ kỹ thuật (Part Drawing, Note, Table) và tích hợp Gemini/GLM để thực hiện OCR chi tiết.

🛠 1. Cài đặt môi trường
Bạn nên sử dụng Python 3.10. Cài đặt các thư viện cần thiết bằng lệnh sau:

Bash
# Cài đặt các thư viện hệ thống (nếu dùng Linux/Docker)
sudo apt-get update && sudo apt-get install -y libgl1 python3-dev build-essential

# Cài đặt thư viện Python
pip install -r requirements.txt
Các thư viện chính: torch, torchvision, gradio, google-genai, transformers, opencv-python-headless.

🚀 2. Huấn luyện Model (Train)
Toàn bộ quy trình huấn luyện được thực hiện trong file faster_rcnn_colab.ipynb.

Dataset: Dữ liệu được gán nhãn theo chuẩn COCO (file instances_default.json).

Kiến trúc: Faster R-CNN với backbone ResNet-50 FPN.

Cách chạy:

Mở file .ipynb trên Google Colab.

Kết nối Google Drive để tải dataset (data.zip).

Chạy các cell để bắt đầu huấn luyện (mặc định 50 epochs).

Kết quả: File weight tốt nhất sẽ được lưu tại /content/best_faster_rcnn_bom.pth.

🔍 3. Chạy Inference Pipeline (Web Demo)
Để chạy giao diện người dùng (Gradio), hãy thực hiện các bước sau:

Đảm bảo file weight faster_rcnn_resnet50_bom.pth (tên mô hình lúc huấn luyện là best_faster_rcnn_bom.pth sau đó được đổi lại )nằm cùng thư mục với app.py.

Thiết lập API Key cho Gemini:

Bash
export GEMINI_API_KEY="your_api_key_here"
hoặc ghi trực tiếp vào hàm GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") trong file app.py
Chạy ứng dụng:

Bash
python app.py
Giao diện sẽ cho phép bạn tải ảnh lên, điều chỉnh ngưỡng tin cậy (Confidence) và xem kết quả detection kèm mã JSON chi tiết.

📊 Báo cáo ngắn
1. Cách tiếp cận 
Detection: Sử dụng Faster R-CNN vì khả năng chính xác cao trong việc xác định các vùng đối tượng nhỏ và phức tạp trong bản vẽ kỹ thuật.

OCR: Tích hợp pipeline kết hợp giữa detection vùng chứa văn bản và sử dụng các mô hình ngôn ngữ lớn (Gemini/GLM) để trích xuất thông tin ngữ cảnh. Vì web demo đang được chạy trên CPU nên tài nguyên bị giới hạng tốc độ khá chậm.

2. Thử nghiệm & Kết quả
Đã thử nghiệm với backbone ResNet-50.

Kết quả tốt nhất: Đạt mAP@0.5:0.95 ~ 0.7449, mAP@0.5 ~ 0.9236 và mAR@100: 0.8136 sau 34 epochs. Loss: 0.0723

<img width="909" height="310" alt="image" src="https://github.com/user-attachments/assets/29b48133-4bec-4d07-bf9f-0cf158d999af" />
Quá trình học hàm loss giảm khá điều
<img width="1113" height="613" alt="image" src="https://github.com/user-attachments/assets/2551d60c-376e-448b-a847-87666040f511" />

Mô hình hoạt động ổn định với các vùng gán nhãn: PartDrawing, Note, Table.
Kết quả test:  <img width="898" height="640" alt="image" src="https://github.com/user-attachments/assets/5fe16546-01aa-476b-a0c4-92bcde49f641" /> <img width="731" height="551" alt="image" src="https://github.com/user-attachments/assets/f3e44823-a1cd-41f5-be0f-c638674adb6e" />




🔗 Tài nguyên
Model Weights: [https://drive.google.com/drive/folders/1TYNrXfxgpcJpLomBqc8EfAB59wXP9fJt?usp=sharing]

Web Demo: [https://huggingface.co/spaces/khac-tuong-222/bom]



# dom
Mã nguồn phát triển hệ thống phát hiện và trích xuất ocr dùng mô hình Faster Faster R-CNN resnet50 ocr dùng mô hình GLM OCR kết hợp API GG studio với mô hình gemini-2.3-flash-lite để sửa lỗi chính tả và thuạt ngữ
