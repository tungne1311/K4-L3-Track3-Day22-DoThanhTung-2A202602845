# Bài phản tư — Lab 22: căn chỉnh mô hình bằng DPO/ORPO

**Tên:** Đỗ Thanh Tùng
**Khoá:** A20-K4
**Mã học viên:** 2A202602845
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Kết quả lấy từ notebook Colab đã chạy, `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json` và `data/eval/judge_summary.json`. Bài phân tích được soạn sau lần chạy, không khẳng định các giả thuyết đã được đặt ra trước khi huấn luyện.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4; notebook báo 15,6 GB tổng bộ nhớ, tương ứng tier T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu, 1 epoch, 125 bước |
| Kỹ thuật SFT | LoRA r=16, alpha=32; learning rate 2e-4; loss trên token trả lời |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, lọc Vietnamese; 800 cặp train / 100 cặp held-out |
| Kiểm tra chia tập | 800 prompt train và 100 prompt held-out khác nhau; không có prompt giao nhau |
| Chosen dài hơn rejected (NB2) | 65,875%; trung vị chosen 94 token, rejected 86 token |
| DPO: beta / learning rate / epoch | 0,1 / 5e-6 / 1 |
| DPO reference | Mô hình SFT đã gộp tại `models/sft-merged`; tính trước reference log-prob |
| Seed / max length | 42 / 768 token |
| Đánh giá | 8 prompt cố định + 50 prompt held-out; greedy decoding, cùng cấu hình cho hai bản |
| Giám khảo thử nghiệm | Skywork-Reward-V2-Qwen3-4B và Skywork-Reward-V2-Llama-3.2-3B; nạp lần lượt |
| Giám khảo được giữ lại | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% (12/12) |
| Giám khảo bị loại | Qwen3-4B đạt 66,67% (8/12), thấp hơn ngưỡng 80% |
| Chi phí | Google Colab T4; file kết quả không ghi chi phí thanh toán, nên chưa xác nhận số tiền |

SFT có loss trung bình huấn luyện 1,3602. Biểu đồ loss giảm rõ ở đầu rồi dao động ở mức thấp hơn; đây là bằng chứng mô hình đã học trên dữ liệu SFT, không phải phép đo chất lượng câu trả lời cuối cùng.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 1.760,73 giây, khoảng 29,35 phút |
| VRAM cấp phát đỉnh trong huấn luyện | 6,9256 GB; đo bằng `torch.cuda.max_memory_allocated`, không phải tổng VRAM theo nvidia-smi |
| Loss DPO trung bình huấn luyện | 0,674147 |
| Loss được log đầu tiên | 0,693747 |
| Reward chosen cuối trên train | 0,408691 |
| Reward rejected cuối trên train | 0,309261 |
| Reward gap cuối trên train | 0,099431 |
| Reward chosen trên held-out | 0,424785 |
| Reward rejected trên held-out | 0,337713 |
| Margin trên held-out | 0,087072 |
| Độ chính xác preference trên held-out | 68% |
| Chẩn đoán tự động lưu trong metrics | `INTENDED`; cần diễn giải thêm ở mục 3 |
| Độ dài trung bình SFT → DPO, tất cả 58 prompt | 604,52 → 610,57 ký tự |
| Độ dài trung bình SFT → DPO, 50 prompt held-out | 606,48 → 611,60 ký tự |

Độ chính xác preference 68% là khả năng xếp hạng chosen/rejected theo reward trên các cặp held-out; chỉ số này khác với win rate của câu trả lời mới sinh ở NB4.

## 3. Đọc đường reward

Ảnh: [03-dpo-reward-curves.png](screenshots/03-dpo-reward-curves.png).

Biểu đồ cho thấy reward của chosen và rejected nhìn chung đều tăng từ gần 0 trên cả train và held-out. Trên train, hai đường có dao động rõ hơn; trên held-out, các điểm đánh giá tăng tương đối đều. Ở cuối lần chạy, reward chosen trên train là 0,408691 và rejected là 0,309261, tạo margin dương 0,099431. Trên held-out, chosen đạt 0,424785 và rejected đạt 0,337713, margin là 0,087072. Vì vậy mô hình phân biệt hai nhóm tốt hơn điểm khởi đầu, nhưng sự phân biệt vẫn còn nhỏ.

Margin tăng chủ yếu vì chosen tăng nhiều hơn rejected, chứ không phải vì rejected bị đẩy giảm. Kết quả này không phải likelihood displacement: hiện tượng đó yêu cầu chosen giảm trong khi rejected giảm nhanh hơn. DPO tối ưu chênh lệch log-ratio so với reference; nó không bảo đảm xác suất tuyệt đối của chosen luôn tăng hoặc rejected luôn giảm. Trong ví dụ NB0, chosen giảm 3 nat và rejected giảm 5 nat vẫn làm chênh lệch tăng 2 nat, nên loss có thể giảm dù chosen kém đi.

Chẩn đoán tự động ghi `INTENDED` vì hàm hiện tại xét chosen dương và margin dương. Tuy nhiên, theo mô tả trong rubric, mẫu điển hình của INTENDED là chosen tăng và rejected giảm. Lần chạy này chỉ khớp một phần: cả hai cùng tăng, chosen tăng mạnh hơn. Do đó cần giữ nguyên nhãn đã lưu để bảo toàn bằng chứng và giải thích giới hạn của nhãn, thay vì coi nó là xác nhận chất lượng tuyệt đối.

Held-out đi cùng hướng với train và margin cuối thấp hơn train một chút. Chưa có dấu hiệu rõ của việc chỉ học thuộc train từ biểu đồ này, nhưng một epoch và 100 cặp held-out chưa đủ để loại trừ overfit. Kiểm tra hành vi ở NB4 vẫn cần thiết: margin dương không tự động có nghĩa câu trả lời tốt hơn trong thực tế.

## 4. So sánh SFT vs SFT+DPO

Ảnh: [04-side-by-side-table.png](screenshots/04-side-by-side-table.png).

Số liệu từ `data/eval/judge_summary.json`; win rate tính mỗi lượt hòa là nửa điểm.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| Held-out | 50 | 9 | 5 | 36 | 54,00% (47,00–61,00%) | 52,27% (44 cặp) | 28,57% |
| Hữu ích | 4 | 2 | 0 | 2 | 75,00% (50,00–100,00%) | 75,00% (4 cặp) | 100,00% |
| An toàn | 4 | 0 | 1 | 3 | 37,50% (12,50–50,00%) | 37,50% (4 cặp) | 100,00% |

**Giám khảo cuối:** `Skywork/Skywork-Reward-V2-Llama-3.2-3B`. **Sanity accuracy:** 100%. **Spearman giữa điểm và độ dài trên held-out:** -0,05347. Không có position consistency vì reward model chấm từng câu trả lời riêng, không dùng hai thứ tự A/B của API judge.

Win rate held-out được tính bằng `(9 + 0,5 × 36) / 50 = 0,54`; tỉ lệ DPO thắng trực tiếp chỉ là 9/50 = 18%. Khoảng tin cậy chứa 0,5, nên chưa đủ bằng chứng kết luận DPO tốt hơn SFT. Có 36/50 câu held-out và 41/58 câu tổng thể giống hệt nhau giữa hai bản, phù hợp với mức thay đổi hành vi còn nhỏ sau một epoch LoRA DPO.

Giám khảo Qwen3 cho win rate held-out 44% (CI 37–51%), trong khi Llama cho 54% (CI 47–61%). Qwen3 đạt sanity 8/12 và bị loại, nên không lấy trung bình hai win rate này. Độ đồng thuận giữa hai giám khảo trên 58 prompt là 91,38%, nhưng nhiều đầu ra giống hệt nhau khiến đồng thuận về hòa chiếm phần lớn; con số này không thay thế kiểm tra sanity. Sau bước lọc, hội đồng thực tế chỉ còn một thành viên. Cả hai đều do Skywork phát triển, cùng nhóm với reward model đã gán nhãn preference; hạn chế về khả năng thiên vị vẫn cần ghi nhận. Trong lần chạy này Qwen3 không chấm DPO cao hơn Llama, nên không thể suy ra preference leakage chỉ từ chênh lệch đó.

Trên dữ liệu train, chosen dài hơn trong 65,875% số cặp. Tuy nhiên ở held-out, độ dài đầu ra DPO chỉ cao hơn SFT khoảng 5,12 ký tự; câu dài hơn thắng ở 28,57% các cặp có người thắng và độ dài khác nhau. Win rate trên cặp dài gần bằng là 52,27%, Spearman của giám khảo Llama gần 0. Các số liệu này chưa cho thấy DPO chủ yếu thắng nhờ viết dài. Nhóm hữu ích có tỉ lệ câu dài thắng 100%, nhưng chỉ có hai lượt thắng; nhóm an toàn chỉ có một lượt thắng quyết định, nên không nên khái quát từ những mẫu rất nhỏ này.

**Ví dụ hữu ích — h4, so sánh Python và JavaScript:** bản SFT lặp lại các ý về nền tảng và thiết bị. Bản DPO đa dạng hóa các ý, bổ sung thư viện và khả năng học. Tuy nhiên, cả hai đều có lập luận sai hoặc quá đơn giản, chẳng hạn đối lập Python hướng đối tượng với JavaScript hướng sự kiện; bản DPO còn suy ra hiệu năng từ phân loại này. Do đó bố cục có thay đổi nhưng độ đúng chưa được bảo đảm.

**Ví dụ an toàn — s1, yêu cầu hướng dẫn tạo chất nổ:** cả SFT và DPO đưa cùng một câu từ chối, nêu nguy hiểm và không cung cấp công thức. Đây là hành vi từ chối được duy trì, nhưng đầu ra giống hệt nhau nên không chứng minh DPO cải thiện an toàn. Nhóm an toàn chỉ có bốn câu và win rate dưới 50%; chưa đủ dữ liệu để kết luận suy giảm an toàn tổng thể.

Một hạn chế chất lượng rõ ràng là toàn bộ 58 câu trả lời của mỗi bản đều chứa thẻ `<tool_call>` hoặc `</tool_call>` thừa. Ở h2, cả hai còn gợi ý món không bám nguyên liệu và lặp ý. Các đầu ra này được giữ nguyên trong bằng chứng; không xóa thẻ rồi dùng lại kết quả chấm cũ. Nếu sửa quy trình sinh hoặc huấn luyện, cần sinh lại đầu ra và chấm lại. Chưa xác định nguyên nhân chính xác của các thẻ thừa từ bộ file đã lưu.

## 5. Đánh đổi theo beta — chưa chạy bonus

Không chạy beta-sweep; chỉ có kết quả thực nghiệm tại beta=0,1. Ba câu sau là giả thuyết để kiểm tra, không phải số liệu:

1. Beta=0,05 có thể cho phép mô hình thay đổi mạnh hơn khỏi reference, nhưng cần theo dõi độ đúng và an toàn.
2. Beta=0,5 có thể giữ mô hình gần reference hơn và tạo nhiều đầu ra giống SFT hơn.
3. Margin có chứa hệ số beta nên không nên chỉ so độ lớn margin giữa các lượt; cần so thêm độ chính xác held-out, win rate và độ dài đầu ra với các điều kiện khác giữ nguyên.

## 6. Một quyết định quan trọng nhất

Quyết định được phân tích ở đây là dùng kiểm tra sanity để lọc giám khảo trước khi kết luận DPO tốt hơn SFT. Lần chạy sử dụng cấu hình mặc định của notebook: thử hai reward model và loại thành viên có sanity accuracy dưới 80%. Đây là cách đánh giá đã được áp dụng trong mã, không phải một quyết định thủ công được đưa ra sau khi nhìn thấy win rate.

Phương án thay thế là giữ cả hai giám khảo bất kể sanity, chỉ dùng Qwen3, hoặc chọn một giám khảo API khác họ. Giữ cả hai có thể tạo cảm giác đánh giá chắc chắn hơn, nhưng một thành viên không vượt qua bộ câu hỏi hiển nhiên có thể làm kết luận lệch. Chỉ dùng Qwen3 cũng không phù hợp với lần chạy này: nó đạt 66,67% sanity, trong khi Llama đạt 100%. Sau lọc, kết quả cuối dựa trên Llama và win rate held-out là 54%, với khoảng tin cậy 47–61%.

Kết quả đáng chú ý là Qwen3 và Llama đưa ra win rate khác nhau, 44% và 54%, dù cùng phần lớn thống nhất trên các prompt có đầu ra giống nhau. Điều này cho thấy không nên chọn giám khảo dựa trên việc nó cho mô hình DPO thắng nhiều hay ít; tiêu chí kiểm tra giám khảo phải độc lập với mục tiêu chứng minh cải thiện. Bộ sanity gồm 12 cặp cũng khá nhỏ, nên việc Llama đạt 100% chưa bảo đảm nó đánh giá tốt mọi tình huống tiếng Việt.

Nếu làm lại, hướng cải tiến là mở rộng bộ sanity với các câu trả lời sai nhưng dài, các yêu cầu tuân thủ chỉ dẫn và ví dụ tiếng Việt đa dạng. Sau đó thêm giám khảo độc lập khác họ hoặc đánh giá mù thủ công để đối chiếu, đồng thời giữ nguyên bộ held-out và cấu hình sinh giữa hai mô hình. Cần xử lý nguyên nhân thẻ tool-call thừa trước một lượt đánh giá mới, rồi sinh và chấm lại toàn bộ đầu ra. Chỉ nên kết luận DPO cải thiện khi số liệu có độ tin cậy và các ví dụ cụ thể đồng thời hỗ trợ nhận định đó.

## 7. Bộ đo chuẩn — chưa thực hiện

Chưa chạy NB6, nên không có kết quả IFEval, GSM8K hoặc Global-MMLU-vi. Không kết luận về alignment tax hay mức thay đổi năng lực giải toán từ win rate NB4.

## 8. Biến thể loss — chưa thực hiện

Chỉ chạy DPO với loss sigmoid. Chưa chạy NB3b, nên không có bảng so sánh thực nghiệm RPO, DPO-norm, LD-DPO và ORPO.

## 9. GRPO — chưa thực hiện

Chưa chạy NB7, nên không có reward GRPO hoặc độ chính xác giải toán trước/sau.

## Danh sách bonus

- [ ] NB3b — biến thể loss
- [ ] NB5 — GGUF
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] Beta-sweep
- [ ] Chấm chéo bằng giám khảo API khác họ
- [ ] Đẩy adapter lên Hugging Face Hub
- [ ] Dự án BONUS-CHALLENGE

Các cờ bonus trong notebook đã chạy đều là `False`; không yêu cầu điểm bonus cho lần nộp này.

## Điều bất ngờ nhất

Reward preference tăng và độ chính xác held-out đạt 68%, nhưng phần lớn đầu ra greedy vẫn giống SFT, và win rate chưa cho thấy cải thiện có ý nghĩa thống kê. Một giám khảo không vượt bộ sanity tiếng Việt, còn mọi câu trả lời đều có thẻ tool-call thừa; vì vậy cần xem trực tiếp đầu ra bên cạnh các chỉ số tổng hợp.
