# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> **

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature thấp, khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng
> nên có câu trả lời nhất quán. Temperature thấp cũng giảm khả năng model tự bịa ra chính sách không có
> thật, và giúp đội QA test được — vì output gần như tái lập được thì mới
> viết được test case. Tôi sẽ để 0.2 thay vì 0.0 để câu chữ mượt mà hơn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với workload này: 10.000 user × 3 call × 350 token = 10,5 triệu token
> output mỗi ngày.
>
> | Model | Giá output /1K | Chi phí/ngày | Chi phí/tháng |
> |---|---|---|---|
> | gpt-4o | $0.010 | $105.00 | $3,150 |
> | gpt-4o-mini | $0.0006 | $6.30 | $189 |
>
> **GPT-4o đắt hơn khoảng 16,7 lần** (tỉ lệ này đúng cho cả input lẫn
> output vì hai mức giá cùng chênh 16,7x) — chênh lệch gần $3.000 mỗi tháng.
>
> Nên dùng GPT-4o: các tác vụ mà một câu trả lời sai tốn nhiều hơn phần
> tiền tiết kiệm được — ví dụ đọc hợp đồng để trích điều khoản, sinh code
> rồi deploy, hay tổng hợp nhiều tài liệu dài cần suy luận nhiều bước. Ở
> đây $3.000/tháng vẫn rẻ hơn một lần sai sót.
>
> Nên dùng mini: các tác vụ khối lượng lớn nhưng đơn giản và dễ kiểm chứng
> — phân loại ý định (intent routing), trả lời FAQ có sẵn, gắn nhãn
> sentiment, tóm tắt đoạn ngắn. Thực tế nên phân tầng: mini xử lý mặc định,
> chỉ đẩy lên 4o khi mini trả về độ tin cậy thấp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Câu trả lời của bạn*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn mình chọn dài 140 từ tiếng Việt:
>
> | | Số từ | tiktoken | Ước lượng từ/0.75 | Lệch |
> |---|---|---|---|---|
> | Tiếng Việt | 140 | 170 tok | 186,7 tok | **−8,9%** |
>
> Ước lượng "số từ / 0.75" **cao hơn** con số thật khoảng 9%. Lý do là công
> thức đó được đặt ra cho tiếng Anh, nơi một từ trung bình dài hơn; còn từ
> tiếng Việt phần lớn là một âm tiết ngắn nên tốn ít token hơn dự đoán.
>
> Về việc tiếng Việt tốn nhiều token hơn tiếng Anh: phải so **cùng nội
> dung** chứ không so cùng số từ, vì 140 "từ" tiếng Việt chỉ tương đương 93
> từ tiếng Anh. Dịch đoạn trên sang tiếng Anh rồi đếm lại:
>
> | Encoding | Tiếng Việt | Tiếng Anh | Tỉ lệ |
> |---|---|---|---|
> | `o200k_base` (GPT-4o) | 170 tok | 106 tok | **1,60x** |
> | `cl100k_base` (GPT-4/3.5) | 302 tok | 108 tok | **2,80x** |
>
> Nguyên nhân: bộ từ vựng BPE được huấn luyện chủ yếu trên văn bản tiếng
> Anh, nên từ tiếng Anh thường nằm gọn trong 1 token còn tiếng Việt bị cắt
> vụn. Chữ có dấu lại là ký tự UTF-8 nhiều byte nên càng dễ vỡ. Ví dụ với
> `cl100k_base`, chữ `nghìn` bị cắt thành 4 token `['n','gh','ì','n']` —
> dấu huyền tách hẳn ra thành một token riêng.
>
> Điểm đáng chú ý: GPT-4o cải thiện rất nhiều cho tiếng Việt — cùng đoạn
> văn giảm từ 302 xuống 170 token, tức **rẻ hơn 44%** chỉ nhờ đổi encoding.
> Cũng chữ `nghìn` đó, `o200k_base` chỉ dùng 3 token `['ng','h','ìn']`.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng đang ngồi chờ trước màn hình và
> câu trả lời dài — chat UI, trợ lý viết bài, giải thích code. Cái nó cải
> thiện không phải tổng thời gian mà là **time-to-first-token**: chữ đầu
> tiên hiện ra sau khoảng nửa giây thay vì màn hình trắng suốt mười giây,
> nên cảm giác nhanh hơn hẳn dù tổng thời gian y nguyên. Thêm nữa, người
> dùng đọc vài dòng đầu là biết model hiểu sai hay chưa và bấm dừng luôn,
> đỡ tốn token. Ngược lại, non-streaming phù hợp khi output là đầu vào cho
> máy chứ không phải cho người: parse JSON, gọi function/tool, chạy batch
> job ban đêm. Những trường hợp đó cần văn bản hoàn chỉnh mới xử lý được,
> mà streaming thì code phức tạp hơn và khó bắt lỗi giữa chừng. Trường hợp
> nữa là khi phải kiểm duyệt nội dung trước khi hiển thị — đã stream ra màn
> hình rồi thì không rút lại được.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Delay cố định giữ nguyên tốc độ gõ cửa server: cứ mỗi giây lại một lần,
> trong khi server đang quá tải cần đúng thứ nó không có là thời gian rảnh
> để xử lý hàng tồn. Exponential backoff giãn dần 0,1s → 0,2s → 0,4s → 0,8s
> nên lưu lượng retry giảm theo cấp số nhân, nhường chỗ cho server hồi
> phục; đồng thời vẫn thử lại nhanh với lỗi chớp nhoáng — mạng chập một
> nhịp thì 0,1 giây sau đã xong.
>
> Nếu hàng nghìn client cùng retry với delay cố định giống nhau thì xảy ra
> **thundering herd**: tất cả bị lỗi gần như cùng lúc, nên cũng retry cùng
> lúc, tạo ra các đợt sóng đồng bộ mỗi giây. Tải lên server không giảm chút
> nào so với lúc đầu, thậm chí tăng vì lượt retry chồng lên request mới —
> server không bao giờ thoát được vòng xoáy (retry storm), một sự cố lẽ ra
> chỉ 5 giây có thể kéo dài vô hạn.
>
> Lưu ý: bản thân exponential backoff **chưa phá được tính đồng bộ** — các
> client vẫn giãn theo cùng một lịch nên vẫn thành sóng, chỉ là sóng thưa
> dần. Muốn giải quyết triệt để phải thêm **jitter** (ngẫu nhiên hóa delay),
> ví dụ `delay = random.uniform(0, base_delay * 2 ** attempt)`. Hàm
> `retry_with_backoff` trong bài chưa có jitter — đây là điểm mình sẽ bổ
> sung nếu đưa vào production.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona mình dùng:
>
> ```
> Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.
> ```
>
> Trợ giảng mặc định người hỏi đang học,
> nên gặp thuật ngữ sẽ giải thích chứ không ném ra rồi đi tiếp. Thử đổi
> thành "chuyên gia AI" thì câu trả lời lập tức dày đặc jargon.
>
> **"trả lời ngắn gọn"** — hai lý do. Về trải nghiệm: đây là CLI, output
> dài thì cuộn mất, không có thanh cuộn đẹp như web. Về chi phí: token
> output đắt gấp 4 lần token input ($0.010 so với $0.0025 cho GPT-4o), nên
> ràng buộc độ dài là đòn bẩy chi phí mạnh nhất — đúng cái mà
> `estimate_cost` trong bài đo được.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất: `history = history[-6:]` — trợ lý chỉ nhớ 3 lượt gần
> nhất.** Người dùng giới thiệu tên ở lượt 1 thì đến lượt 5 model đã quên
> sạch, phải hỏi lại. 
>
> **Cải thiện đề xuất: bộ nhớ tóm tắt lũy tiến (rolling summary).** Thay vì
> vứt bỏ các message cũ, tóm tắt chúng lại rồi giữ bản tóm tắt.
>
> Cách triển khai: thêm biến `summary = ""`. Mỗi khi `len(history) > 6`,
> lấy 2 message sắp bị cắt, gọi một lượt API rẻ bằng `OPENAI_MINI_MODEL`
> với prompt kiểu *"Gộp bản tóm tắt sau với đoạn hội thoại này thành tối đa
> 3 câu, giữ lại tên riêng, con số và các ràng buộc kỹ thuật"*, rồi ghi đè
> `summary`. Khi ghép `messages`, chèn bản tóm tắt thành message system thứ
> hai ngay sau persona:
>
> ```python
> messages = [{"role": "system", "content": persona}]
> if summary:
>     messages.append({"role": "system",
>                      "content": f"Tóm tắt hội thoại trước đó: {summary}"})
> messages += history + [{"role": "user", "content": user_msg}]
> ```
>
> Đánh đổi: mỗi lần cắt tốn thêm một lời gọi.
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
