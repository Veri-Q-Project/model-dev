# PhiUSIIL CSV의 전체 컬럼/타입/샘플값을 한눈에 확인하기 위한 일회성 점검 스크립트.
# 이슈 #3(미사용 feature 통합)에서 어떤 컬럼을 가져올지 결정하는 데에만 사용한다.
import io
import zipfile
import urllib.request

import pandas as pd

PHIUSIIL_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/967/"
    "phiusiil+phishing+url+dataset.zip"
)
USER_AGENT = "Mozilla/5.0 (charcnn-kit dataset downloader)"
HTTP_TIMEOUT = 180


REPORT_PATH = "data/phiusiil_columns_report.txt"


def main():
    print(f"GET {PHIUSIIL_ZIP_URL}")
    req = urllib.request.Request(PHIUSIIL_ZIP_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        blob = resp.read()
    print(f"downloaded {len(blob) / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    lines = []
    lines.append(f"rows={len(df)}, cols={len(df.columns)}")

    if "label" in df.columns:
        lines.append(f"label dist (raw): {df['label'].value_counts().to_dict()}")

    lines.append("")
    lines.append("=== columns ===")
    lines.append(f"{'#':>3} {'name':<28} {'dtype':<10} {'nans':>6} {'nuniq':>7}  sample")
    for i, col in enumerate(df.columns):
        s = df[col]
        sample = s.dropna().head(3).tolist()
        sample_str = ", ".join(repr(x)[:30] for x in sample)
        lines.append(
            f"{i:>3} {col:<28} {str(s.dtype):<10} "
            f"{int(s.isna().sum()):>6} {int(s.nunique()):>7}  {sample_str}"
        )

    if "label" in df.columns:
        num_cols = df.select_dtypes(include="number").columns.drop("label", errors="ignore")
        corr = df[num_cols].corrwith(df["label"]).abs().sort_values(ascending=False)
        lines.append("")
        lines.append("=== |corr(feature, raw_label)| (1=legit, 0=phish) ===")
        for name, v in corr.items():
            lines.append(f"  {v:.4f}  {name}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote report -> {REPORT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
