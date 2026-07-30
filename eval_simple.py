import os
import numpy as np

# ---------- 配置路径 ----------
pred_dir = '/home/wh/code/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_got10k_ep100/got10k'
gt_root = '/mnt/disk2t/datasets/GOT-10k/val'   # 你的验证集根目录
# -----------------------------

def compute_iou(box1, box2):
    """
    box: [x1, y1, w, h]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return inter / (area1 + area2 - inter)

seq_names = [f'GOT-10k_Val_{i:06d}' for i in range(1, 181)]
all_ious = []

for seq in seq_names:
    # 预测文件名带 .txt 后缀
    pred_file = os.path.join(pred_dir, seq + '.txt')
    if not os.path.exists(pred_file):
        print(f"Warning: prediction file missing for {seq}")
        continue
    gt_file = os.path.join(gt_root, seq, 'groundtruth.txt')
    if not os.path.exists(gt_file):
        print(f"Warning: ground truth missing for {seq}")
        continue

    # 读取预测框（假设每行 x,y,w,h，可能带逗号分隔）
    try:
        pred_boxes = np.loadtxt(pred_file, delimiter=',')
    except:
        pred_boxes = np.loadtxt(pred_file)  # 若分隔符不是逗号则尝试空格
    gt_boxes = np.loadtxt(gt_file, delimiter=',')

    # 对齐长度
    min_len = min(len(pred_boxes), len(gt_boxes))
    pred_boxes = pred_boxes[:min_len]
    gt_boxes = gt_boxes[:min_len]

    # 如果预测文件是 Nx4，若格式为 Nx8（带score）则只取前4列
    if pred_boxes.ndim == 2 and pred_boxes.shape[1] > 4:
        pred_boxes = pred_boxes[:, :4]

    seq_ious = [compute_iou(pred_boxes[i], gt_boxes[i]) for i in range(min_len)]
    all_ious.extend(seq_ious)

if not all_ious:
    print("No IoU data collected. Please check paths and file existence.")
else:
    all_ious = np.array(all_ious)
    ao = np.mean(all_ious)
    sr_05 = np.mean(all_ious > 0.5)
    sr_075 = np.mean(all_ious > 0.75)

    print(f"AO: {ao:.4f}")
    print(f"SR0.5: {sr_05:.4f}")
    print(f"SR0.75: {sr_075:.4f}")