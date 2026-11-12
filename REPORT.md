# Báo cáo thực nghiệm SafeRoute

## 1. Bài toán và dữ liệu

Mục tiêu là phân loại mức ảnh hưởng giao thông của một sự cố đã được ghi nhận. Severity 1 là mức ảnh hưởng thấp nhất, Level 4 là cao nhất; đây **không phải** nhãn thương tích/tử vong. Model không dự đoán tai nạn có xảy ra hay không.

Nguồn: [US Accidents (2016–2023) trên Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), dùng bản mẫu 500.000 dòng liên kết bởi tác giả bộ dữ liệu. Trang nguồn nêu giấy phép CC BY-NC-SA 4.0, chỉ cho mục đích nghiên cứu/phi thương mại. Dataset có 500.000 dòng, 49 bang, 9.488 thành phố, từ 2016-01-14 đến 2023-03-31. Phân bố toàn mẫu: Level 1 = 4.274, Level 2 = 398.142, Level 3 = 84.520, Level 4 = 13.064.

Pipeline lấy mẫu phân tầng 60.000 dòng: Level 1 = 513, Level 2 = 47.777, Level 3 = 10.142, Level 4 = 1.568. Chia train/test phân tầng 80/20 với `random_state=42`; test có 12.000 dòng. CV 5-fold chỉ chạy trên train.

Model dùng 16 đặc trưng từ thời điểm bắt đầu, thời tiết, sunrise/sunset, bang, nhiệt độ/độ ẩm/tầm nhìn/gió/mưa và cờ hạ tầng. Tọa độ chỉ dùng trên bản đồ, không đưa vào model. Loại ID, `End_Time`, tọa độ cuối, khoảng cách và mô tả sự cố để tránh leakage. Categorical được impute mode + one-hot; numeric impute median; toàn bộ nằm trong `Pipeline`/`ColumnTransformer` nên fit trong từng fold.

## 2. Lựa chọn thuật toán

- **Majority baseline:** `DummyClassifier(most_frequent)` là mốc đơn giản. Do Level 2 chiếm gần 80%, accuracy baseline cao nhưng macro-F1/balanced accuracy thấp.
- **Decision Tree/CART:** học tương tác phi tuyến trên dữ liệu bảng, không cần scale, dễ giải thích. Gini chọn split; `max_depth=14`, `min_samples_leaf=8`, class weight và `ccp_alpha` chi phối độ phức tạp/mất cân bằng.
- **Random Forest:** gộp phiếu từ bootstrap trees và subset đặc trưng ngẫu nhiên, thường giảm phương sai của cây đơn. Cấu hình: 120 cây, `max_features="sqrt"`, `min_samples_leaf=4`, `class_weight="balanced_subsample"`.

Chọn model bằng CV mean macro-F1; test không dùng để chọn. Random Forest đạt CV macro-F1 cao nhất; nó cũng cho test macro-F1 cao nhất.

## 3. Kết quả

| Mô hình | CV macro-F1 (mean ± std) | CV balanced accuracy (mean ± std) | Train macro-F1 | Test macro-F1 | Test balanced accuracy | Test accuracy | Test weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.2216 ± 0.0000 | 0.2500 ± 0.0000 | 0.2216 | 0.2216 | 0.2500 | 0.7962 | 0.7059 |
| Decision Tree | 0.2540 ± 0.0069 | 0.4088 ± 0.0072 | 0.3263 | 0.2401 | 0.4155 | 0.3719 | 0.4692 |
| Random Forest | **0.3597 ± 0.0109** | **0.4206 ± 0.0133** | 0.6286 | **0.3608** | **0.4293** | 0.6497 | 0.6859 |

Random Forest macro one-vs-rest ROC-AUC trên test: **0.7462**. Baseline accuracy đạt 0.7962 do luôn dự đoán Level 2, nhưng balanced accuracy chỉ 0.25.

| Severity test | Precision | Recall | F1 | Số mẫu |
|---|---:|---:|---:|---:|
| Level 1 · thấp nhất | 0.08 | 0.26 | 0.12 | 103 |
| Level 2 | 0.89 | 0.67 | 0.76 | 9.555 |
| Level 3 | 0.32 | 0.64 | 0.43 | 2.028 |
| Level 4 · cao nhất | 0.11 | 0.14 | 0.12 | 314 |

## 4. Phân tích lỗi và đặc trưng

Model sai 4.204/12.000 mẫu test. Error rate: Level 1 **73,8%**, Level 2 **32,7%**, Level 3 **36,1%**, Level 4 **85,7%**. Hiệu năng ở hai mức hiếm thấp; không nên dùng làm cảnh báo hoặc hỗ trợ quyết định thực tế.

Với nhóm có ít nhất 30 test records, thời tiết Overcast có error rate 0,528 (579 mẫu), bang OH 0,534 (176 mẫu), và ban ngày 0,369 (8.230 mẫu). Đây là mô tả lỗi trên mẫu, không phải quan hệ nhân quả.

Permutation importance cao nhất: `weather_condition` 0,0478, `state` 0,0366, `hour` 0,0267 và `day_of_week` 0,0254 macro-F1 decrease. Random Forest train macro-F1 0,6286, CV 0,3597, gap **0,2689**, cho thấy quá khớp đáng kể. Map và biểu đồ mô tả toàn bộ mẫu 500k; metric model dựa trên stratified 60k.

## 5. Ghi chú vấn đáp

- Gini impurity là `1 - sum(p_k^2)`; entropy là `-sum(p_k log(p_k))`. CART chọn split làm giảm impurity có trọng số ở node con. `max_depth`, `min_samples_leaf`, class weight và `ccp_alpha` điều khiển độ phức tạp.
- Random Forest fit cây trên bootstrap samples, xét subset feature ở mỗi split và gộp phiếu. `n_estimators` tăng độ ổn định/chi phí; `max_features` tăng đa dạng; `min_samples_leaf` regularize cây.
- Macro-F1 cho mỗi lớp trọng số ngang nhau; balanced accuracy là mean recall theo lớp. Accuracy cao không đủ khi nhãn lệch mạnh.
- Bài thực hành 10–15 phút: trong `train_kaggle_accidents.py`, đổi `max_depth=14` thành `8`, chạy `py train_kaggle_accidents.py --sample-size 10000`, rồi đọc lại CV mean/std và test macro-F1. Không dùng test để chọn tham số.

## 6. Giới hạn

Kết quả chịu ảnh hưởng của độ lệch nhãn, nguồn thu thập API, địa lý và thời gian. Severity phản ánh traffic impact trong nguồn Kaggle, không phải severity y tế. Split ngẫu nhiên có thể đánh giá lạc quan hơn triển khai theo thời gian/bang; cần holdout giai đoạn hoặc bang chưa thấy. Dữ liệu được cấp phép phi thương mại; hãy ghi nguồn/citation và không triển khai thương mại.