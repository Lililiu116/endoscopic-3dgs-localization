# evaluations/read_result.py
import sys
import numpy as np
import logging

from evaluations.metrics import summarize_metrics

def q2575(x):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "nan/nan/nan"
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}"

def main(npy_path: str):
    metrics = np.load(npy_path, allow_pickle=True).item()

    # 1) 先打印 summarize_metrics 的“全套输出”（和 benchmark 一样）
    # 确保 logger 能输出（不要 logging.disable）
    logging.getLogger().setLevel(logging.INFO)

    auc_all = summarize_metrics(metrics, auc_thresholds=(1, 5, 10, 15, 50, 100, 500), split='scared')

    # 2) 如果你还想保留你那三行“做表用的简洁输出”，可以继续打印
    rot = q2575(metrics["R_err"])
    trans_cm = q2575(100.0 * np.asarray(metrics["t_err"]))

    auc_str = f"{auc_all[0]:.2f} / {auc_all[1]:.2f} / {auc_all[2]:.2f}"
    

    print("\n--- Table numbers ---")
    print(f"Rotation(°) Q25/50/75:      {rot}")
    print(f"Translation(cm) Q25/50/75:   {trans_cm}")
    print(f"Reproj AUC(%) @1/5/10px:     {auc_str}")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python evaluations/read_result.py xxx.npy"
    main(sys.argv[1])
