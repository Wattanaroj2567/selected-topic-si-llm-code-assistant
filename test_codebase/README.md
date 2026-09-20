# Statistics Pipeline Test Codebase

โฟลเดอร์นี้คือ Python codebase ที่ Local Coding Assistant index และใช้เป็น
Dataset/Resource จริง มี Python 6 files (รวม `__init__.py`) และแยกจาก source
ของระบบหลักอย่างชัดเจน

## Files

- `data_loader.py` — อ่านตัวเลขจาก text file ด้วย `load_numbers`
- `metrics.py` — คำนวณ `mean`, `variance` และ `standard_deviation`
- `reporting.py` — class `StatisticsReport` และ `build_report`
- `pipeline.py` — `run_pipeline` ประสาน loader กับ report builder
- `main.py` — command-line entry point
- `__init__.py` — package marker

## Expected Relationships

Imports:

- `reporting` imports `metrics`
- `pipeline` imports `data_loader` และ `reporting`
- `main` imports `pipeline`

Calls:

- `variance` calls `mean`
- `standard_deviation` calls `variance`
- `build_report` calls `mean` และ `standard_deviation`
- `run_pipeline` calls `load_numbers` และ `build_report`
- `main` calls `run_pipeline`

Class/method:

- `StatisticsReport` contains method `render`

## Run the Dataset Directly

จาก project root:

```powershell
uv run python -m test_codebase.main test_codebase/example_numbers.txt
```

Expected output:

```text
Count: 5 | Mean: 6.00 | Std dev: 2.83
```
