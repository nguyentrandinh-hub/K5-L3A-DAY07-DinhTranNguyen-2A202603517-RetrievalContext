# Báo cáo Day 7 — Agentic RAG

**Sinh viên:** Đinh Trần Nguyên — **MSSV:** `2A202603517`

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Dùng lại bộ corpus 40 trang tài liệu kỹ thuật Gemini API & Antigravity (`data/corpus/*.md`, tài liệu công khai của Google) từ bài Day 6. Sau khi chạy pipeline Day 6 với Strategy B (tối đa 800 ký tự/chunk, trung bình 619.6 ký tự), tập dữ liệu có **2,630 chunks**, được xuất sang `data/day6_converted/chunks.json` thông qua `src/convert_day6.py`.
- **Vector store, embedding model, phiên bản và số chiều:** Kết nối trực tiếp tới PostgreSQL 16 + `pgvector` (`pgvector/pgvector:0.8.7-pg16`, cổng `55433`, database `day6_lab`, bảng `chunks`, lọc `strategy = 'B'`). Model embedding là `openai/text-embedding-3-small` gọi qua OpenRouter, vector **1,536 chiều**, dùng khoảng cách cosine (`<=>`).
- **Agent model, cách chạy và phân chia dev/test:** Agent sử dụng `openai/gpt-4o-mini` qua OpenRouter (`temperature = 0.0`, ép trả JSON action) kết hợp vòng lặp `run_agent` trong `src/agentic.py`. Bộ benchmark gồm **24 câu**, chia làm hai tập trước khi thử nghiệm:
  - **Dev (17 câu):** 15 câu có đáp án (`q01`–`q15`) kế thừa từ Day 6 và 2 câu thiếu bằng chứng (`u01`, `u02`).
  - **Test (7 câu):** 5 câu có đáp án (`q16`–`q20`) kế thừa từ Day 6 và 2 câu thiếu bằng chứng (`u03`, `u04`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** Các câu có đáp án đều có `gold` (`source_id`, `page`, `evidence`) đã được `src/baseline.py` kiểm tra khớp nguyên văn trong chunk. Bốn câu thiếu bằng chứng (`u01`–`u04`) được thiết kế sát chủ đề Gemini API (HIPAA BAA/SOC 2, open weights/VRAM cho Gemini 4 Carbon, uptime SLA/service credit, và pricing theo training steps cho supervised fine-tuning) và đã kiểm tra bằng regex trên toàn bộ 40 tài liệu đều cho **0 kết quả**. Khi chấm test, tôi đọc trực tiếp từng chunk được trích dẫn trong `traces.json` để đối chiếu ngữ nghĩa thay vì chỉ tin vào ID hợp lệ.

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | `tfidf`, `top_k=3`, có filter sẵn nếu có, `budget=800` chars | Mốc lexical ban đầu từ starter | dev | 6/15 (40.0%) | 8/15 (53.3%) | 5/15 (33.3%) | 9.41 ms/câu · \$0 |
| Phương pháp 2 | `bm25` (`k1=1.5, b=0.75`), `top_k=3`, `budget=800` chars | BM25 chuẩn hóa độ dài tài liệu tốt hơn TF-IDF thuần | dev | 11/15 (73.3%) | 13/15 (86.7%) | 9/15 (60.0%) | 8.73 ms/câu · \$0 |
| Phương pháp 3 | `dense` (`pgvector` Day 6, `text-embedding-3-small`), `top_k=3`, `budget=800` chars | Truy xuất ngữ nghĩa bắt được các câu diễn đạt khác từ khóa | dev | 7/15 (46.7%) | 12/15 (80.0%) | 6/15 (40.0%) | 1,334.91 ms/câu · \$0.000007 |
| Phương pháp 4 | `hybrid_rrf` (`dense` + `bm25`, `pool_k=15`, `rrf_k=60`), `top_k=3`, `budget=800` chars | Hợp nhất theo thứ hạng (RRF) để giữ cả từ khóa chính xác lẫn ngữ nghĩa | dev | 11/15 (73.3%) | 12/15 (80.0%) | 8/15 (53.3%) | 1,360.51 ms/câu · \$0.000007 |
| Cải thiện 1 | `hybrid_rrf`, `top_k=3`, tăng `budget=2400` chars | Tăng budget để `pack_context` không làm rơi chunk #2, #3 | dev | 11/15 (73.3%) | 12/15 (80.0%) | **12/15 (80.0%)** | 1,373.07 ms/câu · \$0.000007 |
| Cải thiện 2 | `hybrid_rerank` (Hybrid `pool_k=10` → `gpt-4o-mini` rerank), `top_k=3`, `budget=800` / `2400` | Kỳ vọng LLM rerank đẩy chunk đúng lên top-1 | dev | 7/15 (46.7%) | 10/15 (66.7%) | 6/15 (40.0%) / 10/15 (66.7%) | ~3,072 ms/câu · \$0.00216–0.00366 |
| **Cấu hình đã chọn** | `hybrid_rrf`, `top_k=3`, `budget=2400` chars | Đánh giá sau khi chốt trên dev | **test** | 2/5 (40.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 1,449.73 ms/câu · \$0.000003 |
| Baseline | `tfidf`, `top_k=3`, `budget=800` chars | So sánh đối chứng trên cùng test | **test** | 3/5 (60.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 10.20 ms/câu · \$0 |

### Phân tích lỗi dev, thử nghiệm không cải thiện và lý do chọn cấu hình

Khi đọc kết quả và trace trên tập `dev`, tôi nhận thấy hai lỗi rất rõ ràng:

1. **Lỗi `context_budget` quá chật (`budget=800` ký tự):** Ở cả `bm25`, `dense` lẫn `hybrid_rrf`, `Context Evidence Rate` đều thấp hơn hẳn `Evidence Hit@3` (`dense` tụt từ 12/15 xuống 6/15; `hybrid_rrf` tụt từ 12/15 xuống 8/15). Nguyên nhân là các chunk Strategy B dài trung bình ~620 ký tự, nên `pack_context` chỉ nhét vừa 1 chunk đầu tiên rồi bỏ qua chunk #2 và #3 dù evidence nằm ngay ở đó. Khi tăng `budget` lên `2,400` ký tự (Cải thiện 1), `Context Evidence Rate` của `hybrid_rrf` tăng thẳng từ **8/15 (53.3%) lên 12/15 (80.0%)**, giải quyết triệt để lỗi rơi evidence mà không tốn thêm chi phí API. Trên Agent `dev`, thay đổi này giúp giảm số câu trả lời bị từ chối nhầm từ **7/15 xuống 0/15** (trả lời đúng đủ 15/15 câu có đáp án, trung bình chỉ cần 1.06 lượt search thay vì 2.29 lượt ở Baseline).
2. **Thử nghiệm `hybrid_rerank` bị phản tác dụng (báo cáo trung thực kết quả không cải thiện):** Tôi thử dùng `gpt-4o-mini` xếp hạng lại top-10 candidate từ Hybrid RRF, nhưng kết quả lại **tụt** từ `Hit@1 = 11/15, Hit@3 = 12/15` xuống chỉ còn `7/15` và `10/15`, trong khi thời gian tăng hơn gấp đôi (~3.1 giây/câu) và chi phí tăng hàng trăm lần (`$0.0022–0.0037` cho 15 câu). Soi lại code cho thấy `LlmRerankRetriever` chỉ gửi 450 ký tự đầu mỗi chunk cho reranker để tiết kiệm token, vô tình cắt mất phần bảng biểu hoặc câu chốt nằm ở nửa sau chunk khiến model đánh giá sai độ liên quan.

Vì vậy, theo tiêu chí chốt trước khi chạy `test` (ưu tiên `Context Evidence Rate` cao, độ trễ vừa phải và chi phí thấp), tôi chọn **`hybrid_rrf` + `budget=2400` + `max_searches=4` + `top_k=3`** thay cho cả Baseline lẫn `hybrid_rerank`.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Agent dùng một tool tìm kiếm duy nhất (thay đổi retriever theo cấu hình: `TfidfRetriever` cho Baseline và `HybridRrfRetriever` cho cấu hình đã chọn), cho phép gọi tối đa `max_searches = 4` lượt, mỗi lượt lấy `top_k = 3` chunks.
  - Ở mỗi vòng lặp, `pack_context` loại trùng theo `chunk_id` và gom các chunk vào `budget` ký tự (800 ở Baseline, 2,400 ở cấu hình chọn). Agent dừng khi model trả action `answer` kèm citation hợp lệ (`[chunk-id]` nằm trong context và trong văn bản trả lời), hoặc tự động `abstain` nếu hết 4 lượt search, context rỗng, hoặc thiếu citation.
  - Mọi lỗi kết nối PostgreSQL / OpenRouter đều không dùng fallback giả; nếu phát sinh ngoại lệ trong `run_eval.py`, hệ thống ghi `status: "error"` kèm thông báo lỗi vào `traces.json` (thực tế cả `dev` và `test` đều ghi nhận **0 lỗi**).

| Cấu hình / split | Tool use | Số lượt gọi (TB) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---:|---|---:|---:|---:|---:|
| Baseline (`tfidf`, `budget=800`) / `test` | 7/7 | 2.43 | 4 / 0 / 1 \| **0.40** | 1/1 | 1/1 | 2/2 | 4/5 |
| Selected (`hybrid_rrf`, `budget=2400`) / `test` | 7/7 | 1.86 | 1 / 1 / 3 \| **1.40** | 6/7 | 4/4 | 2/2 | 1/5 |

*(Trên tập `dev` gồm 17 câu, Baseline trả lời được 8/15 câu có đáp án và từ chối nhầm 7/15 câu, chi phí \$0.004625; cấu hình Selected trả lời đủ 15/15 câu có đáp án và từ chối đúng 2/2 câu thiếu bằng chứng, chi phí \$0.004247).*

### Chi tiết chấm thủ công từng câu `test` (cấu hình Selected: `hybrid_rrf`, `budget=2400`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt search (`gemini-doc-29::B::0072`, `gemini-doc-09::B::0003`, `gemini-doc-02::B::0010`) | Trả lời đúng model là `Gemini 4 Hydrogen` `[gemini-doc-29::B::0072]`, nêu trung thực chưa có giá Hydrogen trong context và dẫn giá Argon `[gemini-doc-09::B::0003]` | **Grounded = 1**, **Citation = 2/2**. Context thiếu chunk giá ở doc 07 nên chỉ đúng một phần; 2 citation đều khớp nguồn. |
| `q17` | Có đáp án | 1 lượt search (`gemini-doc-37::B::0002`, `0006`, `0005`) | Giải thích đúng khác biệt giữa `rate_limit_exceeded` và `quota_exceeded` (cùng mã 429) `[gemini-doc-37::B::0006][gemini-doc-37::B::0005]` | **Grounded = 2**, **Citation = 1/2**. Câu trả lời đúng và đủ, nhưng đọc đối chiếu cho thấy `0005` là bảng lỗi `404/409` chứ không chứa `rate_limit_exceeded` (nằm ở `0002` cũng có trong context). |
| `q18` | Có đáp án | 1 lượt search (`gemini-doc-20::B::0066`, `0001`, `0076`) | Từ chối trả lời (`abstain`), không có citation | **Grounded = 0**, **Từ chối nhầm**. Retrieval lấy đúng tài liệu `doc-20` nhưng bỏ lỡ đoạn nói về file TXT/Markdown/HTML. |
| `q19` | Có đáp án | 1 lượt search (`gemini-doc-01::B::0005`, `0024`, `0013`) | Trả lời đúng fallback về `llms.txt` (`https://aistudio.google.com/llms.txt`) `[gemini-doc-01::B::0005]` | **Grounded = 2**, **Citation = 1/1**. Đối chiếu `0005` thấy câu gốc khớp hoàn toàn. |
| `q20` | Có đáp án | 1 lượt search (`gemini-doc-27::B::0004`, `gemini-doc-31::B::0087`, `27::B::0003`) | Trả lời đúng mặc định là unrestricted outbound network và hạn chế bằng `network` allowlist `[gemini-doc-27::B::0004][gemini-doc-31::B::0087]` | **Grounded = 2**, **Citation = 2/2**. Cả hai chunk đều hỗ trợ trực tiếp câu trả lời. |
| `u03` | Thiếu bằng chứng | 4 lượt search (`gemini-doc-05::B::0019`, `0054`) | Từ chối (`abstain`) sau khi tìm không thấy thông tin SLA / service credit | **Abstention đúng**. Không bịa thêm con số uptime. |
| `u04` | Thiếu bằng chứng | 4 lượt search (`gemini-doc-08::B::0004`, `10::B::0031`, `08::B::0024`) | Từ chối (`abstain`) sau 4 lượt tìm về fine-tuning Gemini 4 Carbon | **Abstention đúng**. Tuân thủ đúng ngưỡng thiếu bằng chứng. |

*(Đối chiếu với Baseline trên `test`: chỉ trả lời được mỗi `q20` ở mức 2 điểm, từ chối nhầm cả 4 câu `q16`–`q19` và từ chối đúng `u03`–`u04`).*

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Code: `src/agentic.py`, `src/baseline.py`, `src/convert_day6.py`, `src/agent_metrics.py`, `src/run_eval.py`.
  - Dữ liệu chuẩn hóa: `data/day6_converted/chunks.json`, `questions.json`, `agent_cases.json`.
  - Kết quả và trace `dev`: `results/dev-tfidf/`, `results/dev-bm25/`, `results/dev-dense/`, `results/dev-hybrid-rrf-800/`, `results/dev-hybrid-rrf-2400/`, `results/dev-hybrid-rerank-800/`, `results/dev-hybrid-rerank-2400/`, `results/dev-agent-baseline/`, `results/dev-agent-hybrid-2400/`.
  - Kết quả, trace và nhãn `test`: `results/test-tfidf/`, `results/test-hybrid-rrf-2400/`, `results/test-agent-baseline/` (`traces.json`, `reviews.json`, `agent_metrics.json`), `results/test-agent-hybrid-2400/` (`traces.json`, `reviews.json`, `agent_metrics.json`).
  - Lệnh chạy tái lập chính:
    ```bash
    python -X utf8 -m unittest discover -s tests -v
    python -X utf8 src/run_eval.py --mode retrieval --method hybrid_rrf --split test --budget 2400 --output results/test-hybrid-rrf-2400
    python -X utf8 src/run_eval.py --mode agent --method hybrid_rrf --split test --budget 2400 --max-searches 4 --top-k 3 --model openai/gpt-4o-mini --output results/test-agent-hybrid-2400
    python -X utf8 src/agent_metrics.py --cases data/day6_converted/agent_cases.json --traces results/test-agent-hybrid-2400/traces.json --reviews results/test-agent-hybrid-2400/reviews.json --split test --output results/test-agent-hybrid-2400/agent_metrics.json
    ```
  - Tổng thời gian & chi phí trên `test`: Baseline tốn median `5,248.23 ms/câu`, chi phí **\$0.001751**; Selected tốn median `6,397.88 ms/câu`, chi phí **\$0.002539** (cộng thêm \$0.000003 cho bước retrieval).

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước và sau:** Chuyển từ `tfidf` (`budget=800`) sang `hybrid_rrf` (`budget=2400`) giúp điểm trung bình Grounded answer trên `test` tăng từ **0.40 lên 1.40** (gấp 3.5 lần) và giảm tỷ lệ từ chối nhầm từ **4/5 xuống 1/5**, trong khi vẫn giữ **2/2** abstention chính xác trên các câu thiếu bằng chứng.
  - **Đánh đổi:** `hybrid_rrf` chậm hơn `tfidf` ở khâu truy xuất (~1.4s so với ~10ms vì phải gọi API embedding và query pgvector) và `budget=2400` làm prompt dài hơn một chút (~12.9k so với ~9.0k prompt tokens trên 7 câu test). Tuy nhiên trên thực tế, do tìm trúng evidence ngay ở lượt đầu nên tổng chi phí trên `dev` thậm chí còn rẻ hơn Baseline (\$0.004247 vs \$0.004625).
  - **Giới hạn:** (1) Tập `test` chỉ có 7 câu nên mỗi câu chiếm tỷ trọng lớn (chỉ cần 1 câu như `q18` trượt top-3 là tỷ lệ tụt 20%); (2) Việc đọc đối chiếu ở `q17` cho thấy model vẫn có thể gắn nhầm ID của một chunk lân cận trong cùng tài liệu dù câu trả lời đúng — kiểm tra ID tự động trong code không phát hiện được lỗi này; (3) Với câu thiếu bằng chứng (`u03`, `u04`), Agent có xu hướng search đủ cả 4 lượt trước khi chịu từ chối, làm tăng độ trễ và token.

## Thử thách nâng cao (nếu thực hiện)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** OpenAI *GPT-4 Technical Report* ([arXiv:2303.08774](https://arxiv.org/abs/2303.08774), bản công khai trên arXiv), đúng **100 trang** (`data/advanced/gpt4_technical_report.pdf`, 5.0 MB), gồm text học thuật, 21 trang chứa bảng benchmark/exam, 8 trang chứa ảnh nhúng (multimodal examples) và 23 trang chứa biểu đồ (`Figure`).
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Dùng `pymupdf` trích xuất text toàn bộ **100 trang** (0 trang bị bỏ trống), chia thành **426 text chunks** (`max_chars=900`, `overlap=150`, giữ `page` gốc).
  - Với **36 trang** có bảng, ảnh nhúng hoặc `Figure`, hệ thống render ảnh PNG (`150 DPI`, lưu tại `data/advanced/pages/page_XXX.png`) và dùng `openai/gpt-4o-mini` Vision tạo **27 visual-summary chunks** (mô tả cấu trúc bảng, trục/chú giải/đơn vị biểu đồ và chi tiết ảnh). Toàn bộ **453 chunks** được embed bằng `openai/text-embedding-3-small` (1,536 chiều). Chi phí index: **\$0.151241** (`335.67s`).
  - Pipeline Agent (`src/advanced_rag.py`) trang bị 2 tool: `search` (Hybrid RRF trên cả text và visual summary, `top_k=6`, `budget=8000` chars) và `inspect_page(page)` (mở trực tiếp ảnh PNG trang gốc bằng Vision để kiểm chứng ô bảng, trục biểu đồ và ảnh trước khi trả lời). Phần tài liệu tham khảo/danh sách tác giả dài ở cuối PDF vẫn được index text đầy đủ, không cần xử lý ảnh.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - Bộ `dev` gồm 18 câu (`adv-d01`–`adv-d15` bao phủ đủ 5 nhóm + `adv-u01`–`adv-u03` thiếu bằng chứng) đạt **15/15 trả lời đúng citation + 3/3 abstain đúng** (`\$0.00987`).
  - Ở lần chạy test đầu (`data/advanced/benchmark.json`), pipeline đạt **26/30 (86.7%)** do bỏ qua `inspect_page` ở 2 câu bảng. Theo đúng quy tắc của `ADVANCED.md`, sau khi chỉnh sửa (`top_k=6`, `budget=8000`, bắt buộc `inspect_page` với câu hỏi bảng/biểu đồ/ảnh), tôi tạo bộ **30 câu `test` mới hoàn toàn độc lập** (`data/advanced/benchmark_v2.json`, gồm 24 câu có đáp án + 6 câu thiếu bằng chứng) để đánh giá lại.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---:|---:|---|
| Bảng | 5 / 6 | 83.3% | Đạt `t25, t26, t27, t29, t30`; sai `t28` (đọc nhầm cột 84.2% LLaMA few-shot thay vì 85.6% ALUM) |
| Ảnh, sơ đồ | 5 / 5 | 100.0% | Đạt toàn bộ `t31`–`t35` (ảnh VGA/Lightning, chart 1997, sơ đồ nhiệt tiếng Pháp, Extreme Ironing, InstructGPT) |
| Biểu đồ | 5 / 5 | 100.0% | Đạt toàn bộ `t36`–`t40` (đúng trục, đơn vị, legend và giá trị matrix Figure 11) |
| Multi-hop | 2 / 4 | 50.0% | Đạt `t41, t43`; `t42` từ chối nhầm; `t44` thiếu nửa sau ở trang 63 |
| Text | 4 / 4 | 100.0% | Đạt toàn bộ `t45`–`t48` |
| Thiếu bằng chứng | 6 / 6 | 100.0% | Từ chối chính xác toàn bộ `u10`–`u15` |
| **Toàn bộ** | **27 / 30** | **90.0%** | **Đạt ngưỡng ≥ 90% (0 lỗi hệ thống)** |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:**
  - Trace và nhãn chấm đối chiếu bản gốc từng câu được lưu tại `results/advanced-test-v2/traces.json`, `results/advanced-test-v2/summary.json` và `results/advanced-test-v2/reviews.json`.
  - Thời gian median: **10,062.0 ms/câu**; tổng chi phí chạy 30 câu test: **\$0.082144**; lỗi hệ thống: **0**.
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - **Giới hạn:** Câu multi-hop dài (`t42`, `t44`) đôi khi không kích hoạt đủ lượt `search`/`inspect_page` cho trang thứ hai; bảng nhiều cột dày đặc như Table 2 (`t28`) vẫn có rủi ro đọc nhầm cột cạnh nhau nếu không crop sát từng hàng.
  - **Đường dẫn & lệnh chạy tái lập:**
    ```bash
    python -X utf8 src/advanced_rag.py --mode index
    python -X utf8 src/advanced_rag.py --mode run --split dev --output results/advanced-dev
    python -X utf8 src/advanced_rag.py --mode run --benchmark data/advanced/benchmark_v2.json --split test --output results/advanced-test-v2
    ```
