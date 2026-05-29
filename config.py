"""
EchoRisk-MICCAI 2026 全局配置文件

使用方法:
    from config import DATA_ROOT, TASK1_ROOT
"""

# ============================================================
# 数据集根目录 (dicom/ 和 labels/ 的上一级目录)
# ============================================================
DATA_ROOT = r"D:\玛卡巴卡\MATERIALS\22spring\EchoRisk竞赛\Dataset"

# ============================================================
# 任务一路径
# ============================================================
TASK1_ROOT = DATA_ROOT + r"\Task1"
TASK1_DICOM_ROOT = TASK1_ROOT + r"\dicom"
TASK1_LABELS_ROOT = TASK1_ROOT + r"\labels"

TASK1_TRAIN_DICOM = TASK1_DICOM_ROOT + r"\train"
TASK1_VAL_DICOM = TASK1_DICOM_ROOT + r"\val"

TASK1_TRAIN_CSV = TASK1_LABELS_ROOT + r"\task1_train.csv"
TASK1_VAL_CSV = TASK1_LABELS_ROOT + r"\task1_val.csv"

# ============================================================
# 实验输出目录
# ============================================================
OUTPUT_ROOT = r"output"
OUTPUT_SEGMENTATION = OUTPUT_ROOT + r"\segmentation"
OUTPUT_VIDEO = OUTPUT_ROOT + r"\video"

# ============================================================
# 任务二 / 任务三路径 (预留)
# ============================================================
# TASK2_ROOT = DATA_ROOT + r"\Task2"
# TASK3_ROOT = DATA_ROOT + r"\Task3"
