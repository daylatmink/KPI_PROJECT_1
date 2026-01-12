"""
GIẢI THÍCH: Cách Cost từ Bước 6.5 Được Sử Dụng Trong Bước 7
============================================================

## QÚTRÌNH TRUYỀN DỮ LIỆU

### Bước 6.5: Tạo Cost Profile
- Script: 06b_assign_cost_to_assignees.py
- Input: data/interim/assignee_skill_profile.csv (skill của mỗi assignee)
- Output: data/interim/assignee_cost_profile.csv
  
  Columns:
  - Assignee (A-1, A-2, ...)
  - hourly_cost_usd (10.0 - 140.5)
  - seniority_level (junior, mid, senior, expert)
  - num_skills, total_tasks, ...

### Bước 6: Xuất Assignee Profiles (CẬP NHẬT ĐỂ THÊM COST)
- Script: 06_export_assignee_profiles.py
- Đọc: assignee_cost_profile.csv từ bước 6.5
- Output: projects/ZOOKEEPER/assignees.csv
  
  Columns (NEW):
  - assignee_code: A-1, A-2, ...
  - assignee_id: UUID
  - skills: "Bug fixing / Maintenance;Build & Release Engineering"
  - skill_scores: "23;8;1;6;7;4;5;30"
  - hourly_cost_usd: 77.0, 70.0, 18.0, ... ✨ THÊM MỚI
  - seniority_level: senior, junior, ... ✨ THÊM MỚI

### Bước 7: Sử Dụng Cost Trong Assignment Algorithms
- Scripts: 07_hs_topo_assign.py, 07_ihs_topo_assign.py, 07_ghs_topo_assign.py, 07_mohs_topo_assign.py

## CHI TIẾT CÁC BƯỚC TRONG BƯ ƠC 7

### Step 1: Load Cost Data
```python
# Dòng 452 trong 07_hs_topo_assign.py
emp_costs = load_assignee_costs(assignees_path)
# Trả về: {"A-1": 77.0, "A-10": 70.0, "A-100": 18.0, ...}
```

### Step 2: Truyền Cost Vào Context
```python
# Dòng 479
ctx = BatchContext(batch_df, emp_skills, emp_costs)
#                                          ^^^^^^^^^ THÊM MỚI
```

BatchContext lưu trữ:
```python
self.emp_costs = emp_costs or {}  # {"A-1": 77.0, "A-10": 70.0, ...}
```

### Step 3: Sử Dụng Cost Trong Scoring

ObjectiveWeights cập nhật để thêm cost_optimization:
```python
@dataclass
class ObjectiveWeights:
    skill_matching: float = 0.55        # 55%
    workload_balance: float = 0.15      # 15%
    priority_respect: float = 0.15      # 15%
    skill_development: float = 0.05     # 5%
    cost_optimization: float = 0.10     # 10% ✨ MỚI
```

Score() method gọi 5 hàm scoring (thay vì 4):
```python
def score(self, assign: Dict[str, str]) -> Tuple[float, Dict]:
    s1 = self._skill_matching(assign)        # 55%
    s2 = self._workload_balance(assign)      # 15%
    s3 = self._priority_respect(assign)      # 15%
    s4 = self._skill_dev(assign)             # 5%
    s5 = self._cost_optimization(assign)     # 10% ✨ MỚI
    
    total = (
        s1 * 0.55 + s2 * 0.15 + s3 * 0.15 + s4 * 0.05 + s5 * 0.10
    )
    
    return total, {
        "skill_matching": s1,
        "workload_balance": s2,
        "priority_respect": s3,
        "skill_development": s4,
        "cost_optimization": s5,  # ✨ MỚI
        "total": total,
    }
```

### Step 4: Hàm _cost_optimization() Chi Tiết

```python
def _cost_optimization(self, assign):
    """
    Tính cost optimization score từ 0 đến 1
    - Mục tiêu: Cân bằng cost giữa các nhân viên
    - Score cao = distribution cost tốt
    """
    if not self.data.emp_costs:
        return 1.0  # Không có data, cho điểm trung lập
    
    # Bước 1: Tính cost cho mỗi task
    emp_costs = {}  # {assignee_code: total_cost}
    
    for task_id, assignee_code in assign.items():
        # Lấy info task
        info = self.data.get_task_info(task_id)
        duration = info.get("Duration_Hours", 1.0)  # Ví dụ: 5.5 giờ
        
        # Lấy hourly rate từ emp_costs
        hourly_rate = self.data.emp_costs.get(assignee_code, 50.0)
        # Ví dụ: A-1 → 77.0 $/hr
        
        # Tính cost của task này
        task_cost = hourly_rate * duration
        # Ví dụ: 77.0 * 5.5 = 423.5 $
        
        # Cộng vào total của nhân viên
        emp_costs[assignee_code] = emp_costs.get(assignee_code, 0) + task_cost
    
    # Bước 2: Tính cân bằng cost
    # Ví dụ: emp_costs = {"A-1": 950$, "A-10": 820$, "A-100": 540$}
    
    cost_values = list(emp_costs.values())  # [950, 820, 540]
    cost_mean = np.mean(cost_values)        # 770
    cost_std = np.std(cost_values)          # 195
    
    # Bước 3: Tính coefficient of variation
    cost_ratio = cost_std / cost_mean      # 195 / 770 = 0.253
    
    # Bước 4: Chuyển thành score (0-1)
    # Ratio thấp = distribution tốt = score cao
    balance_score = 1.0 / (1.0 + cost_ratio)
    # = 1.0 / (1.0 + 0.253) = 0.798
    
    return balance_score
```

## VÍ DỤ THỰC TẾ

### Scenario: Gán 3 task cho 3 nhân viên

**Input Data:**
- Task 1: 5 giờ, cần Bug fixing (difficulty 2)
- Task 2: 3 giờ, cần Testing (difficulty 1)
- Task 3: 4 giờ, cần Features (difficulty 3)

- A-1: $77/hr, giỏi tất cả skills
- A-10: $70/hr, giỏi tất cả skills
- A-100: $18/hr, giỏi Bug fixing/Testing

**Phương án 1: Không xem xét cost**
- Assign Task 1 (5h) → A-1 ($77): $385
- Assign Task 2 (3h) → A-10 ($70): $210
- Assign Task 3 (4h) → A-100 ($18): $72
- Total: $667
- Cost distribution: [385, 210, 72] → Std: 147, Mean: 222.3
- cost_ratio = 147/222.3 = 0.66
- balance_score = 1/(1+0.66) = 0.60

**Phương án 2: Xem xét cost (ưu tiên cân bằng)**
- Assign Task 1 (5h) → A-100 ($18): $90
- Assign Task 2 (3h) → A-100 ($18): $54
- Assign Task 3 (4h) → A-1 ($77): $308
- Total: $452
- Cost distribution: [308, 0, 144] → Std: 124, Mean: 151
- cost_ratio = 124/151 = 0.82
- balance_score = 1/(1+0.82) = 0.55

Nhưng với Harmony Search optimization:
- Assign Task 1 (5h) → A-100 ($18): $90
- Assign Task 2 (3h) → A-10 ($70): $210
- Assign Task 3 (4h) → A-1 ($77): $308
- Total: $608
- Cost distribution: [308, 210, 90] → Std: 94, Mean: 203
- cost_ratio = 94/203 = 0.46
- balance_score = 1/(1+0.46) = 0.68 ✨ TỐT HƠN

## IMPACT TRÊN KẾT QUẢ

### Trước Cập Nhật:
- Chỉ xem xét: Skill match, Workload balance, Priority, Skill dev
- Có thể gán task khó cho senior dev chuyên môn → Chi phí cao

### Sau Cập Nhật:
- Thêm cost optimization vào scoring (10% trọng số)
- Cân bằng cost distribution + skill match + priority
- Kết quả: 
  ✅ Tasks phân phối hợp lý theo skill
  ✅ Workload cân bằng
  ✅ Cost cân bằng giữa các nhân viên
  ✅ Có thể tính toán tổng chi phí project

## OUTPUT BƯỚC 7

File: projects/ZOOKEEPER/hs_score.json

```json
{
  "batch_1": {
    "task_1": {"assignee": "A-100", "score": 0.82},
    "task_2": {"assignee": "A-10", "score": 0.79},
    "task_3": {"assignee": "A-1", "score": 0.85}
  },
  "summary": {
    "total_batches": 3,
    "average_score": 0.82,
    "cost_info": {
      "A-1": {"assigned_tasks": 5, "total_hours": 22, "total_cost": 1694},
      "A-10": {"assigned_tasks": 4, "total_hours": 18, "total_cost": 1260},
      "A-100": {"assigned_tasks": 2, "total_hours": 8, "total_cost": 144}
    }
  }
}
```

## SUMMARY

Cost từ bước 6.5 được tích hợp vào pipeline:

6.5 (Generate Cost)
   ↓ assignee_cost_profile.csv
   ↓
6 (Export Profiles) - Thêm cost columns
   ↓ projects/ZOOKEEPER/assignees.csv (với hourly_cost_usd)
   ↓
7 (Assignment)
   ├→ Load emp_costs
   ├→ Pass to BatchContext
   ├→ Use in _cost_optimization() scoring
   └→ Influence final assignment + reduce total cost
"""
