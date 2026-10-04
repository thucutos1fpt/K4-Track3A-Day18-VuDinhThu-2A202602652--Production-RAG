# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Vũ Đình Thu  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng

| Lecture concept | Module | Hàm cụ thể | Quan sát ngắn |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Tách theo độ tương đồng giữa các câu giúp hạn chế cắt giữa ý. Khi model chưa sẵn sàng, fallback vẫn giữ pipeline chạy được. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 tốt với từ khóa, số tiền và tên chính sách; dense search hỗ trợ các cách hỏi khác nhau. RRF giúp kết hợp hai nguồn mà không cần chuẩn hóa điểm. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank top candidates trước khi tạo câu trả lời giúp giảm context không liên quan. Đây là bước hữu ích nhất cho các câu hỏi ngắn và dễ bị lẫn chủ đề. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Kết quả thật: Faithfulness 0.7369, Relevancy 0.7077, Precision 0.9208, Recall 0.7167. Precision cao nhưng recall và grounding vẫn cần cải thiện. |
| Contextual embeddings | M5 | `contextual_prepend()` | Thêm tên/ngữ cảnh tài liệu vào chunk giúp embedding biết đoạn văn thuộc policy nào, nhất là với những đoạn chỉ có số liệu. |

## Phần 2: Khó khăn và cách giải quyết

- **Lỗi gặp phải:** `ModuleNotFoundError: No module named 'qdrant_client'` khi chạy `python src/pipeline.py` bằng Python hệ thống.
  - **Cách xử lý:** Kiểm tra lại interpreter và chạy bằng `.venv\Scripts\python.exe`. Bài học là cài dependency xong chưa đủ; cần chắc chắn lệnh chạy dùng đúng virtual environment.

- **Lỗi gặp phải:** `ModuleNotFoundError: No module named 'pypdf'` khi test phần load tài liệu.
  - **Cách xử lý:** Cài dependency trong `.venv` và thêm xử lý fallback để PDF không làm dừng cả pipeline. Các PDF scan vẫn cần OCR nếu muốn dùng làm context.

- **Điều mình rút ra:** Các policy có version cũ/mới như phép năm và mật khẩu rất dễ gây trả lời sai nếu chỉ search theo nội dung. Lần sau mình sẽ thêm metadata `version` và `status=active/superseded` từ đầu thay vì chỉ dựa vào prompt.

## Phần 3: Action plan cho project cá nhân

### Project: Trợ lý hỏi đáp tài liệu nội bộ

#### Hiện trạng

- Pipeline đã có chunking, hybrid search, reranking và đánh giá RAGAS.
- Điểm yếu hiện tại là câu hỏi multi-hop, câu hỏi phủ định và policy có nhiều version.

#### Kế hoạch cải tiến

1. **Chunking:** Dùng hierarchical chunking để retrieve child nhỏ nhưng vẫn trả parent đủ ngữ cảnh.
2. **Search:** Giữ Hybrid Search + RRF; BM25 ưu tiên số tiền, ngưỡng và tên policy.
3. **Reranking:** Dùng CrossEncoder cho top-20 và chỉ đưa top-3 vào prompt.
4. **Evaluation:** Chạy RAGAS định kỳ trên test set cố định; theo dõi Faithfulness và Recall trước.
5. **Enrichment:** Bổ sung metadata về category, version và trạng thái hiệu lực của policy.

#### Timeline

- **Tuần 1:** Chuẩn hóa metadata/version, chạy lại test set và ghi nhận baseline mới.
- **Tuần 2:** Tối ưu prompt có trích dẫn context, phân tích các câu dưới 0.7 và đo latency từng bước.
