import pandas as pd
from pathlib import Path

# Đường dẫn tới thư mục chứa dữ liệu gốc và thư mục lưu dữ liệu đã xử lý
raw_data_dir = Path("../data/raw")
processed_data_dir = Path("../data/processed")

# Tạo thư mục lưu dữ liệu đã xử lý nếu chưa tồn tại
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Load và xử lý dữ liệu training
def load_and_process_train_data():
    train = pd.read_csv(
        raw_data_dir / "train.csv",
        usecols=["store_nbr", "family", "date", "sales", "onpromotion"],
        dtype={
            "store_nbr": "category",
            "family": "category",
            "sales": "float32",
            "onpromotion": "uint32"
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    train["date"] = train["date"].dt.to_period("D")
    train = train.set_index(["date", "family", "store_nbr"]).sort_index()
    train.to_csv(processed_data_dir / "train_processed.csv")
    return train

# Load và xử lý dữ liệu test
def load_and_process_test_data():
    test = pd.read_csv(
        raw_data_dir / "test.csv",
        dtype={
            "store_nbr": "category",
            "family": "category",
            "onpromotion": "uint32",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    test["date"] = test["date"].dt.to_period("D")
    test = test.set_index(["date", "family", "store_nbr"]).sort_index()
    test.to_csv(processed_data_dir / "test_processed.csv")
    return test

# Load và xử lý dữ liệu holidays/events
def load_and_process_holidays_events():
    holidays_events = pd.read_csv(
        raw_data_dir / "holidays_events.csv",
        dtype={
            "type": "category",
            "locale": "category",
            "locale_name": "category",
            "description": "category",
            "transferred": "bool",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    holidays_events["date"] = holidays_events["date"].dt.to_period("D")
    holidays_events = holidays_events.set_index("date")
    holidays_events.to_csv(processed_data_dir / "holidays_events_processed.csv")
    return holidays_events

# Gọi các hàm xử lý và lưu trữ dữ liệu đã xử lý
if __name__ == "__main__":
    train = load_and_process_train_data()
    test = load_and_process_test_data()
    holidays_events = load_and_process_holidays_events()
    print("Dữ liệu đã được xử lý và lưu vào thư mục 'processed'")
