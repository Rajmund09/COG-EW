import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "cogew_rf_cnn_test_evaluation.json"
)


def describe(value, indent=0):
    prefix = " " * indent

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                print(f"{prefix}{key}: dict ({len(item)} keys)")
                describe(item, indent + 2)
            elif isinstance(item, list):
                print(f"{prefix}{key}: list ({len(item)} items)")
                if item:
                    first = item[0]
                    if isinstance(first, (dict, list)):
                        print(f"{prefix}  first item:")
                        describe(first, indent + 4)
                    else:
                        print(f"{prefix}  sample: {first!r}")
            else:
                print(f"{prefix}{key}: {item!r}")

    elif isinstance(value, list):
        print(f"{prefix}List length: {len(value)}")
        if value:
            describe(value[0], indent + 2)


def main():
    if not REPORT.exists():
        raise FileNotFoundError(f"Metrics report not found: {REPORT}")

    with REPORT.open("r", encoding="utf-8") as file:
        report = json.load(file)

    print("COG-EW TEST METRICS STRUCTURE")
    print("=" * 45)
    describe(report)


if __name__ == "__main__":
    main()
