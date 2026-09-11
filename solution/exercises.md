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
> *Khi temperature 0.0 câu trả lời tương đối ổn định, trực tiếp và ít có sự thay đổi giữa các lần gọi. Khi tăng temperature lên 0.5,1.0,1.5 có sự đa dạng trong cách diễn đạt câu trả lời  *

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> * Em sẽ chọn temperature 0.5 cho chatbot hỗ trợ khách hàng, để có thể thống nhất về cách trong phong cách trả lời cho khách hàng, các temperature càng cao không thống nhất được dự diễn đạt giữa các câu trả lời  *

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> * Tổng số token đầu ra mỗi ngày là `10.000 × 3 × 350 = 10.500.000 token`. Theo bảng giá của GPT-4o có giá output 0,01 USD/1.000 token, tương đương khoảng 105 USD/ngày; GPT-4o-mini có giá 0,0006 USD/1.000 token, tương đương khoảng 6,3 USD/ngày. Như vậy xét riêng output, GPT-4o đắt hơn khoảng `105 / 6,3 ≈ 16,7 lần`. GPT-4o phù hợp với các tác vụ phức tạp như phân tích tài liệu quan, cần đưa ra suy luận nhiều bước hoặc xử lý yêu cầu khó; GPT-4o-mini phù hợp với công việc trả lời nhanh, phân loại yêu cầu, tóm tắt ngắn hoặc chatbot có số lượng request lớn để tối ưu chi phí. *

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *  Với persona giáo viên tiểu học, phản hồi sử dụng từ ngữ đơn giản, câu ngắn và thường dùng ví dụ trực quan, chẳng hạn ví blockchain như một cuốn sổ được nhiều người cùng giữ. Với persona chuyên gia tài chính, phản hồi dài và chuyên sâu hơn, sử dụng các thuật ngữ như distributed ledger, consensus, cryptographic hash, decentralization và smart contract. Mặc dù câu hỏi của người dùng giống hệt nhau, system prompt đã thay đổi mức độ chi tiết, từ vựng, ví dụ và phong cách diễn đạt của model. Điều này cho thấy system prompt có thể được dùng để định hình vai trò và hành vi của trợ lý cho từng nhóm người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Với một đoạn tiếng Việt khoảng 100 từ, cách ước lượng của Part 1 cho khoảng `100 / 0,75 ≈ 133 token`, trong khi khi đếm bằng tokenizer số token thực tế có thể cao hơn đáng kể; trong lần thử của tôi kết quả vào khoảng 160 token, tức cao hơn khoảng `(160 - 133) / 133 × 100 ≈ 20%`. Nguyên nhân là token không tương ứng trực tiếp với một từ: một từ tiếng Việt có thể được tokenizer tách thành nhiều token, đặc biệt với dấu, âm tiết hoặc những cụm từ ít xuất hiện trong dữ liệu huấn luyện. Vì vậy việc dùng tokenizer thực tế đáng tin cậy hơn cách ước lượng dựa trên số từ.


---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất đối với các ứng dụng hội thoại hoặc tác vụ tạo câu trả lời dài, vì người dùng có thể nhìn thấy nội dung ngay khi model bắt đầu sinh thay vì phải chờ toàn bộ câu trả lời hoàn thành. Điều này làm giảm cảm giác chờ đợi và cải thiện trải nghiệm dù tổng thời gian xử lý có thể không giảm. Ngược lại, non-streaming phù hợp hơn khi chương trình cần nhận toàn bộ kết quả trước khi xử lý tiếp, ví dụ phân loại văn bản, sinh JSON để chương trình khác đọc, gọi API backend tự động hoặc các tác vụ batch không có người dùng chờ trực tiếp.
*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff làm thời gian chờ tăng dần sau mỗi lần thất bại, ví dụ 0,1 giây → 0,2 giây → 0,4 giây, nên khi server đang quá tải client sẽ giảm dần tần suất gửi request và cho hệ thống thời gian phục hồi. Nếu hàng nghìn client đều retry với delay cố định giống nhau, chúng có thể gửi lại request gần như cùng thời điểm, tạo thành một đợt tải mới và tiếp tục làm server quá tải, hiện tượng này thường được gọi là “thundering herd”. Exponential backoff giúp giảm áp lực lên API và làm hệ thống ổn định hơn; trong hệ thống thực tế có thể bổ sung thêm random jitter để tránh nhiều client retry đồng thời.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona tôi chọn là: **"Bạn là trợ giảng thân thiện của khóa AI, giải thích chính xác, dễ hiểu, trả lời ngắn gọn bằng tiếng Việt và đưa ví dụ khi cần."** Tôi dùng cụm "trợ giảng thân thiện" để định hướng phong cách giao tiếp gần gũi với người đang học. Yêu cầu "trả lời ngắn gọn bằng tiếng Việt" giúp câu trả lời tập trung vào vấn đề và phù hợp với người dùng Việt Nam; còn "đưa ví dụ khi cần" giúp những khái niệm kỹ thuật khó hiểu trở nên trực quan hơn.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý hiện tại là history chỉ giữ ba lượt hội thoại gần nhất, vì vậy khi cuộc trò chuyện dài hơn model có thể quên những thông tin quan trọng ở các lượt trước. Tôi sẽ cải thiện bằng cách bổ sung cơ chế memory: các message gần nhất vẫn được giữ nguyên, còn phần hội thoại cũ được tóm tắt thành một bản summary ngắn và đưa summary đó vào context ở những lượt sau. Với hệ thống lớn hơn, có thể lưu thông tin cần nhớ vào database hoặc vector database và chỉ truy xuất những nội dung liên quan đến câu hỏi hiện tại. Cách này vừa duy trì được ngữ cảnh dài hạn vừa tránh để số token đầu vào tăng không giới hạn.
*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
