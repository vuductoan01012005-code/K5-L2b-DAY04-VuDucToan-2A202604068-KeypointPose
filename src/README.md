# Source code

Lab này làm trên CVAT ở máy bạn. Phần lớn việc chạy lệnh đã gói trong `lab.py` ở thư mục gốc:

| Lệnh | Làm gì |
|---|---|
| `python lab.py check` | Tìm CVAT (Docker Desktop, Docker Engine, hoặc Docker trong WSL), bật nếu đang tắt, sửa lỗi OPA và lỗi kẹt "Connecting…", báo RAM / ổ đĩa |
| `python lab.py setup --mssv <MSSV> --ten <HoVaTen>` | Tải bộ ảnh của bạn vào `data/`, tạo project `d04-keypoint` (label lấy từ `src/lab/labels_d04_keypoint.json`), tạo task, giao job cho tài khoản của bạn, ghi `src/exports/assignment.json` |
| `python lab.py export` | Xuất COCO Keypoints 1.0 của các task vào `src/exports/`, rồi chạy `check_submission.py` |
| `python lab.py lite` / `full` / `stop` / `start` | Tiết kiệm RAM cho máy yếu (xem `docs/MAY_YEU.md`) |

Đăng nhập: `lab.py` hỏi tên đăng nhập và mật khẩu CVAT mỗi lần cần (không lưu). Muốn khỏi gõ lại, tạo Access token trong CVAT (avatar → Profile → Access tokens) rồi đặt biến môi trường `CVAT_TOKEN`.

Thư mục `src/` còn chứa script tự kiểm, schema hai label và nơi đặt file xuất.

- File bắt đầu chạy: `src/check_submission.py`
- Cách chạy (từ thư mục gốc repo, Python 3.9+, không cần cài thư viện):

  ```bash
  python src/check_submission.py --persons src/exports/HoVaTen_persons.zip --faces src/exports/HoVaTen_faces.zip
  ```

  Bonus B1 / B2: kiểm riêng file bonus, đổi số ảnh kỳ vọng:

  ```bash
  python src/check_submission.py --persons src/exports/HoVaTen_bonus_persons.zip --expected-person-images 3
  python src/check_submission.py --faces src/exports/HoVaTen_bonus_faces.zip --expected-face-images 5
  ```

- Input: file **COCO Keypoints 1.0** (`.zip` hoặc file `.json` bên trong) xuất từ task `persons-<tên>` và `faces-<tên>`, không kèm ảnh.
- Output: các dòng `[LỖI]` / `[CẢNH BÁO]` theo từng ảnh, rồi phần **TÓM TẮT** (số ảnh, số skeleton theo ảnh, tỉ lệ cờ `v = 2 / 1 / 0`, `num_keypoints` trung bình) và dòng `Kết quả: ĐẠT` hoặc `Kết quả: CHƯA ĐẠT`. Mã thoát 1 nếu còn `[LỖI]`.
- File học viên được sửa: không cần sửa script hay `keypoint_schema.json`. Bạn chỉ đặt 2 file xuất vào `src/exports/`.

## Script kiểm gì

| Kiểm tra | Mức | Gợi ý lỗi tương ứng |
|---|---|---|
| Zip có kèm ảnh | LỖI | Quên bỏ chọn Save images |
| Không có file COCO trong zip, không có label cần kiểm | LỖI | Sai định dạng xuất, xuất nhầm task |
| Tên điểm của label khác bảng của lab (thiếu, thừa, sai chính tả) | LỖI | Label sai: báo coach |
| Mảng `keypoints` không đủ 51 số (người) / 39 số (mặt), `v` ngoài 0 / 1 / 2 | LỖI | Sai định dạng xuất |
| `v = 0` nhưng `x, y` khác 0 | Ghi chú | Không phải lỗi: CVAT 2.x giữ tọa độ cũ cho điểm `outside`; điểm `v = 0` bị bỏ qua khi huấn luyện và chấm |
| `num_keypoints` khác số điểm `v > 0` | LỖI | Định dạng |
| Điểm `v > 0` nằm ngoài ảnh | LỖI | `vis_flag`: điểm ngoài ảnh phải là `outside` |
| Skeleton `face` trong file của task persons (hoặc ngược lại) | LỖI | Vẽ nhầm label |
| Tên điểm đúng nhưng sai thứ tự, số cạnh khác 19 / 13 | CẢNH BÁO | Label lệch: báo coach |
| Điểm `v > 0` nằm sát mép ảnh (≤ 1 px) | CẢNH BÁO | Kéo điểm ngoài ảnh về mép (`vis_flag`) |
| Mọi điểm đều `outside` (`num_keypoints = 0`) | CẢNH BÁO | Skeleton thừa (`extra`) |
| Người có các điểm trải cao dưới 60 px | CẢNH BÁO | Người quá nhỏ (`extra`), hoặc người bị cắt khung |
| Một cặp trái/phải nằm ngược phía so với các cặp còn lại của cùng người | CẢNH BÁO | `lr_swap` một cặp, hoặc tay / chân bắt chéo thật |
| Thấy rõ mũi và hai mắt nhưng các điểm left nằm bên trái ảnh | CẢNH BÁO | `lr_swap` cả người (gán như quay lưng) |
| Mặt có `left_eye_*` bên trái ảnh; khóe miệng ngược phía hai mắt | CẢNH BÁO | `lr_swap` khuôn mặt |
| Khóe ngoài gần sống mũi hơn khóe trong; mí trên nằm dưới mí dưới | CẢNH BÁO | `position` / đặt nhầm điểm |
| Hai skeleton gần như trùng nhau | CẢNH BÁO | Hai skeleton cho một người (`extra`) |
| Ảnh người không có skeleton; ảnh mặt không đúng 1 skeleton | CẢNH BÁO | `missing` / `extra` |
| Số ảnh khác số ảnh được giao (12 / 8) | CẢNH BÁO | Xuất nhầm task |

Script **không** có ground truth nên không tính OKS và không biết điểm có nằm đúng khớp hay không. Không có cảnh báo chưa chắc là đúng: vẫn phải tự review (CP3). Có cảnh báo cũng chưa chắc là sai: đọc từng cảnh báo, sửa nếu đúng là lỗi, hoặc ghi lý do giữ nguyên vào báo cáo.

## `keypoint_schema.json`

Tên điểm, cạnh và cặp trái/phải của hai label `person` (17 điểm COCO, kèm σ của COCO) và `face` (13 điểm của lab). Script đọc file này. Bảng dễ đọc nằm ở `docs/GUIDELINE.md`.

## Thư mục `exports/`

Đặt đúng hai file (xem `SUBMISSION.md`):

```text
src/exports/HoVaTen_persons.zip
src/exports/HoVaTen_faces.zip
```
