# REPORT — Lab Day 10: Qwen3-VL-4B Road Damage Localization

**Họ tên:** Bùi Đức Thành
**Bài học:** VinUni AI20k — Track 4 — Day 10
**Dataset version:** `d10-21e4070301`
**Đánh giá:** `test` gồm 311 ảnh; micro-F1 tại IoU ≥ 0.5, bootstrap CI95, 1.000 lượt lấy mẫu.
**Mã lệnh:** `lab_day10.ipynb`; không sửa `scripts/` và `tests/` do đề cung cấp.

## Bảng tổng hợp thí nghiệm

| Mã | Setup | Tham số train | Peak VRAM (GB) | Thời gian train (phút) | Micro F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | e0 | 0 | — | — | 0.0898 | [0.0631, 0.1178] | 0.929 | 0.1146 | 0.0272 | 0.2066 |
| E1 | e1 | 33,030,144 | 6.75 | 38.67 | 0.2767 | [0.2383, 0.3176] | 1.270 | 0.2010 | 0.1985 | 0.5364 |
| E2 | e2 | 142,135,296 | 8.44 | 41.31 | 0.3473 | [0.3100, 0.3860] | 1.109 | 0.2381 | 0.2833 | 0.5827 |
| E3 | e3 | 148,426,752 | 8.37 | 54.43 | 0.4113 | [0.3720, 0.4510] | 1.006 | 0.2929 | 0.3421 | 0.6481 |
| E4 (r=4) | rank 4 | 118,935,552 | 7.94 | 57.52 | 0.3919 | [0.3527, 0.4308] | 1.186 | 0.2745 | 0.3251 | 0.6502 |
| E4 (r=64) | rank 64 | 266,391,552 | 10.17 | 53.91 | 0.4268 | [0.3859, 0.4690] | 0.994 | 0.3629 | 0.3322 | 0.6855 |

*Các con số lấy nguyên từ `report/evidence/*.json`, được chấm bằng `scripts/score.py` với `coord=norm1000`. Thời gian và VRAM là số đo trong từng phiên Colab, do đó không nên diễn giải như benchmark phần cứng chính xác giữa các phiên khác nhau. E0 không có huấn luyện.*

## Câu 1 — Kiến trúc và khối được cập nhật

Qwen3-VL-4B gồm mô hình ngôn ngữ xấp xỉ 4,0 tỷ tham số (bao gồm embedding), vision transformer (ViT) khoảng 415 triệu và các tầng nối thị giác–ngôn ngữ: `visual.merger` ~27 triệu và `visual.deepstack_merger_list` ~82 triệu tham số, tổng chính xác **109.105.152** tham số projector theo lần kiểm tra `named_parameters()` của notebook. Cấu trúc ViT trích đặc trưng từ ảnh, projector đưa đặc trưng về không gian mà LLM tiếp nhận, và LLM autoregressive sinh danh sách hộp bao JSON.

- **E1:** LoRA trên 252 linear module thuộc self-attention và MLP của LLM, train **33.030.144** tham số; đóng băng ViT và projector.
- **E2:** Giữ LLM LoRA và cập nhật đầy đủ projector ở FP32; tổng **142.135.296** tham số.
- **E3:** Thêm LoRA trên 96 linear module thuộc attention/MLP của ViT (tổng 348 module LoRA); cập nhật **148.426.752** tham số.

Base LLM được nạp NF4 4-bit; ViT và `lm_head` không lượng tử hóa.

## Câu 2 — E0 zero-shot và hai lỗi thường gặp

E0 có `parse_valid_rate` **84.89%** (264/311), F1 **0.0898**, precision **0.1242** và recall **0.0703**. Hai vấn đề nổi bật trong output và số đo là:

1. **JSON dài bị cắt / lặp hộp:** 47/311 output không parse hợp lệ theo bộ chấm. Một output thô của `image_id=vlcsnap-2025-02-18-23h07m47s847` lặp nhiều `crack` và kết thúc giữa một tọa độ, ví dụ đoạn cuối: `..., {"label":"crack","bbox_2d": [350, 250, 400, 275], "confidence": 0.7}, {"label":"crack","bbox_2d": [350, 225,` — không đóng danh sách JSON.
2. **Bỏ sót đối tượng / localization sai:** E0 recall chỉ **7.03%**, khiến số false negative cao (**542 FN** trên 583 box GT). Việc trả lời được JSON không đồng nghĩa localization chính xác; crack F1 chỉ **0.0272**.

Mẫu JSON hợp lệ từ E0 ở `image_id=vlcsnap-00200`: `[{"label":"manhole","bbox_2d": [770, 640, 843, 687]}]` (hợp lệ định dạng, nhưng phải so với GT mới biết đúng box hay không).

## Câu 3 — Thang tọa độ 0–1000 so với pixel

Tọa độ được mã hóa theo kích thước tương đối của ảnh (`x_norm1000 = x_pixel/width*1000`, tương tự y). Ground truth của bộ chấm dùng thang 0–1000, nên đưa số 0–1000 vào chế độ `--coord pixel` sẽ khiến script **chia thêm** cho width/height và làm hộp dịch vị trí, đổi kích thước; IoU sai và F1 giảm. Với ảnh 640×360, cùng tọa độ số không mô tả cùng một vùng giữa hai hệ quy chiếu.

**Kiểm chứng trực tiếp trên cùng predictions E3:** chấm E3 trên cùng 311 ảnh với `coord=norm1000` cho F1 **0.4113**, nhưng dùng nhầm `coord=pixel` cho F1 **0.0000** (chênh lệch **-0.4113**). Đây là thử nghiệm đổi cách *diễn giải output* khi chấm, không phải train lại model.

## Câu 4 — Assistant-only loss masking

Nhãn training lấy `input_ids` từ prompt + output, còn `labels` gán `-100` cho mọi token trước câu trả lời, kể cả system/user prompt và image tokens. Loss cross-entropy chỉ chạy trên JSON đầu ra của assistant. Nếu tính trên prompt và token ảnh, tối ưu sẽ bị phân tán vào việc tái tạo phần input đã biết, làm méo mục tiêu dự đoán hộp. Kết quả smoke test notebook cho một sample: **566 token tổng**, **316 token mask**, **250 token assistant**; câu trả lời có ký hiệu kết thúc `<|im_end|>`, vision grid `[[1,22,40]]` tương ứng **220 vision token**.

## Câu 5 — Quantization và FP32 projector

Trong E2/E3, `visual.merger` và `visual.deepstack_merger_list` cần full fine-tuning. Nếu thay bằng các layer 4-bit lượng tử hóa, tham số không còn ở dạng floating-point `nn.Parameter` có thể cập nhật đầy đủ bằng AdamW; cần đưa khối projector về precision có thể huấn luyện. Notebook nạp base bằng NF4 cho LLM và bỏ qua `model.visual`, `lm_head` khi quantize; projector được cast **FP32 trước khi gắn LoRA** và `requires_grad_(True)` sau khi PEFT đóng băng base. Đã kiểm chứng **24 tensors**, **109.105.152** tham số projector ở FP32 và có gradient hữu hạn. FP32 giúp giảm nguy cơ không ổn định gradient khi train full projector trong khi phần lớn backbone ở fp16/NF4.

## Câu 6 — Chia train/val/test theo session

Các ảnh được lấy từ video dashcam, nhiều frame lân cận gần như trùng nội dung. Chia từng ảnh ngẫu nhiên dễ để frame gần giống xuất hiện ở cả train và test, làm rò rỉ đặc trưng ngữ cảnh và khiến F1 test lạc quan giả. Bộ script gộp theo `session` (ngày/giờ quay) rồi phân cả session vào một split, không trùng nhóm giữa train/val/test; notebook đã xác nhận **13 train sessions, 2 val sessions, 3 test sessions**, giao ba tập rỗng.

| Split | Ảnh | Pothole boxes | Crack boxes | Manhole boxes |
|---|---:|---:|---:|---:|
| Train | 1.422 | 1.074 | 1.750 | 691 |
| Val | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |

Sự mất cân bằng giữa các split, nhất là val chỉ có 66 pothole box, nhấn mạnh yêu cầu xem F1 từng lớp chứ không chỉ micro F1.

## Câu 7 — Projector và LoRA ViT có xứng đáng?

E1 F1 **0.2767**, E2 **0.3473**, E3 **0.4113**; mở projector giúp **+0.0706** F1, thêm ViT LoRA giúp **+0.0641** F1. Chi phí training từ E1 **38.67 phút / 6.75 GB** sang E2 **41.31 phút / 8.44 GB**, E3 **54.43 phút / 8.37 GB**. Nếu VRAM bị giới hạn, E1 là phương án rẻ; nếu cần localization tốt hơn, E2/E3 cho kết quả vượt trội. E3 có F1 crack **0.3421**, tăng so với E1 **0.1985**.

Các CI95 E2 **[0.3100, 0.3860]** và E3 **[0.3720, 0.4510]** còn chồng lấn, vì vậy không thể tuyên bố mọi cải thiện nhỏ đều có ý nghĩa thống kê chỉ bằng CI riêng lẻ.

## Câu 8 — 311 ảnh test và giới hạn suy luận

Dùng **1.000 lần bootstrap theo ảnh** để tính khoảng tin cậy 95% cho micro-F1, giữ phân cụm box của mỗi ảnh trong cùng lượt resampling. CI E3 là **[0.3720, 0.4510]**, độ rộng **0.0791**. E4 rank 64 có CI **[0.3859, 0.4690]**, độ rộng **0.0831**. Hai khoảng chồng lấn nhiều; riêng việc một F1 điểm cao hơn chưa cho phép khẳng định thắng về mặt tổng quát. Muốn kết luận mạnh cần kiểm định paired bootstrap giữa các output trên cùng ảnh, mở rộng dữ liệu độc lập theo session, hoặc báo kết quả trên các seed khác nhau.

## Câu 9 — Phân tích lỗi theo lớp và ảnh minh họa

Ở setup E4 rank 64, **crack** có F1 thấp nhất (**0.3322**), tiếp theo pothole (**0.3629**), cao nhất manhole (**0.6855**). Crack có đường biên dài/mảnh, dễ phân đoạn thành box nhỏ hoặc lẫn với texture mặt đường; dataset có nhiều box nhỏ (median area theo README khoảng 0,7% ảnh). Đây là các giả thuyết phù hợp mô hình lỗi; cần đối chiếu ảnh để khẳng định từng trường hợp.

Minh họa thực nghiệm sẽ ở `report/figures/compare_01.png` ... `compare_06.png`, mỗi ảnh gồm GT bên trái và prediction bên phải, chọn trên test dựa trên TP/FP/FN và IoU 0,5. Chạy `python finish_day10_cpu.py` để tạo sáu ảnh kèm chú thích nguồn/box chính xác. Xem `report/figures/README.md`.

## Câu 10 — E4 rank ablation

Giữ kiến trúc E3 (LLM LoRA + ViT LoRA + full FP32 projector), dataset, seed 42, lr 2e-4, batch 2, gradient accumulation 4, 300 optimizer steps, 0.05 LoRA dropout và quy ước alpha=2r. Chỉ đổi **LoRA rank**: r=4 so với r=64, và đối chiếu r=16 của E3.

- r=4: **0.3919** F1, CI95 **[0.3527, 0.4308]**, **118.935.552** tham số train, peak **7.94 GB**.
- r=16: **0.4113** F1, CI95 **[0.3720, 0.4510]**, **148.426.752** tham số train, peak **8.37 GB**.
- r=64: **0.4268** F1, CI95 **[0.3859, 0.4690]**, **266.391.552** tham số train, peak **10.17 GB**.

r=64 hơn r=4 **+0.0349** điểm F1 và hơn r=16 **+0.0155**. Tuy nhiên CI95 chồng lấn rõ và quá trình chạy trên Colab gặp giới hạn quota, runtime bị reset; chưa có đủ cơ sở kết luận tăng rank thật sự tạo cải thiện nhất quán ngoài tập test. Trade-off chắc chắn là số tham số và VRAM tăng (r64 hơn r4 **2.23 GB** peak theo hai run đã đo).

## Nhìn lại

Thách thức lớn nhất không chỉ là train mô hình mà là **tái lập thí nghiệm**: runtime Colab bị giới hạn GPU giữa training/inference, phải backup PEFT adapter **và** projector full-train riêng, dùng optimizer state để resume khi còn training. `torchao` phiên bản cũ từng gây lỗi khi PEFT attach LoRA; sau khi gỡ package xung đột, LoRA/projector restore thành công. Ở inference, checkpoint step 300 đã giúp tránh train lại model; predictions `.jsonl` phải được backup định kỳ để không mất kết quả khi Colab ngắt. Các con số test cũng cho thấy JSON hợp lệ 100% không đồng nghĩa bắt đúng vật thể: crack vẫn có F1 thấp hơn manhole đáng kể. Một hướng phát triển là kiểm định theo session và mở rộng dữ liệu crack nhỏ, nhưng chưa thực hiện trong lab này.
