# Nguồn ảnh của bài lab (`src/lab/images/`)

- **Ảnh người** (`000000xxxxxx.jpg`, 140 ảnh): COCO 2017 train/val (Lin et al., *Microsoft COCO: Common Objects in Context*, 2014). Mỗi ảnh là ảnh Flickr với giấy phép CC BY 2.0 hoặc CC BY-SA 2.0; giấy phép và đường link Flickr gốc của từng ảnh nằm trong `pool.json` (trường `license`, `source`).
- **Ảnh khuôn mặt** (`face_<image_id>.jpg`, 90 ảnh): cắt quanh khuôn mặt và phóng to từ ảnh COCO 2017 có cùng `image_id`, giấy phép CC BY 2.0 / CC BY-SA 2.0 như ảnh gốc; link gốc trong `pool.json` (trường `page`).

Ảnh có giấy phép CC BY-SA được chia sẻ lại theo cùng giấy phép. Chỉ dùng trong lớp; khi đưa ảnh chụp màn hình vào báo cáo, ghi nguồn "COCO 2017".

Đáp án (tọa độ điểm) **không** nằm trong repo này.
