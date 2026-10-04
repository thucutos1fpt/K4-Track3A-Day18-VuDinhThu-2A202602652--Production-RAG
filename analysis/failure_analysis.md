# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** [Họ và tên]  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | N/A | 0.7369 | N/A |
| Answer Relevancy | N/A | 0.7077 | N/A |
| Context Precision | N/A | 0.9208 | N/A |
| Context Recall | N/A | 0.7167 | N/A |

RAGAS evaluated 20 questions using the configured OpenAI API. Context precision
is strongest; faithfulness, answer relevancy, and context recall are the main
improvement opportunities.

## Bottom-5 Failures

### Run-specific analysis

The pipeline completed 20 queries and produced `reports/ragas_report.json`.
The following five items are the lowest aggregate RAGAS results.

| Question | Expected evidence | Likely failure mode | Suggested fix |
|---|---|---|---|
| Nghỉ không lương 20 ngày | CEO approval; employee pays own insurance after 14 days | Faithfulness 0.4429: generated claim insufficiently grounded | Require answer citations and retrieve the approval threshold chunk |
| Thử việc có phép năm? | No; unpaid leave needs manager approval | Faithfulness 0.5000: negative policy is easy to hallucinate | Add explicit negative-policy instruction and cite source |
| Thử việc có bảo hiểm PVI? | No; only mandatory social insurance | Faithfulness 0.5000: benefit eligibility evidence is missing/weak | Retrieve employment-status chunk and enforce answer grounding |
| Junior probation salary ceiling | 17,000,000 VND/month | Context recall 0.5765: salary-band evidence not retrieved | Expand candidate top-k and boost numeric/salary terms in BM25 |
| Senior, 9 years: leave + salary | 18 leave days and 20–35m VND/month | Answer relevancy 0.5833: multi-hop response incomplete | Use query decomposition for leave and salary before generation |

**Error-tree case study — annual leave:**

1. Output correct? It must answer CEO approval and the insurance condition.
2. Context correct? It must contain both the 16–30 day approval threshold and the >14-day insurance rule.
3. Query rewrite/metadata correct? Preserve “20 ngày”, “không lương”, and “phê duyệt” as high-value terms.
4. Fix location: retrieval and answer-generation grounding, before final output.

### #1
- **Question:**
- **Expected:**
- **Got:**
- **Worst metric:**
- **Error Tree:** Output sai → Context đúng? → Query OK? →
- **Root cause:**
- **Suggested fix:**

### #2
(copy template)

### #3
(copy template)

### #4
(copy template)

### #5
(copy template)

## Case Study (cho presentation)

**Question chọn phân tích:**

**Error Tree walkthrough:**
1. Output đúng? →
2. Context đúng? →
3. Query rewrite OK? →
4. Fix ở bước:

**Nếu có thêm 1 giờ, sẽ optimize:**
-
