import numpy as np
import cv2
from skimage import io, color, measure, morphology, feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
import geopandas as gpd

# =========================
# 必要输入（请按需修改）
# =========================
image_path = "your_image.png"        # 原始影像，灰度或RGB
gt_mask_path = "your_gt_mask.png"    # 二值真值：1=异常人工表面，0=正常
impervious_mask_path = None          # 可选：人工表面掩膜(1=人工表面)；若无则为全图

# 对象生成与评估参数
min_object_area = 50                 # 连通域最小像素数
closing_size = 3                     # 形态学闭运算核大小
asai_threshold = 0.0                 # ASAI 判别阈值（经验上 >0 视为异常）

# RF 参数（与文中设置思路一致：多树、特征随机子采样等）
rf_params = dict(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1)

# =========================
# 工具函数
# =========================
def to_gray(img):
    if img.ndim == 3:
        return color.rgb2gray(img)
    img = img.astype(np.float32)
    return img / (img.max() if img.max() > 1 else 1.0)

def norm01(arr, eps=1e-12):
    m, M = np.nanmin(arr), np.nanmax(arr)
    if M - m < eps:
        return np.zeros_like(arr)
    return (arr - m) / (M - m)

def glcm_props(img_u8, mask=None, win=7, levels=256, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    滑窗 GLCM 六特征：Variance, Second Moment(ASM), Dissimilarity, Contrast, Homogeneity, Entropy
    返回 (H,W,6)
    """
    H, W = img_u8.shape
    pad = win // 2
    pad_img = np.pad(img_u8, pad, mode="reflect")
    feats = np.zeros((H, W, 6), np.float32)

    for y in range(H):
        for x in range(W):
            patch = pad_img[y:y+win, x:x+win]
            glcm = feature.greycomatrix(patch, distances=distances, angles=angles,
                                        levels=levels, symmetric=True, normed=True)
            contrast = feature.greycoprops(glcm, 'contrast').mean()
            dissim  = feature.greycoprops(glcm, 'dissimilarity').mean()
            homo    = feature.greycoprops(glcm, 'homogeneity').mean()
            asm     = feature.greycoprops(glcm, 'ASM').mean()  # second moment
            # 熵
            p = glcm.astype(np.float64)
            entropy = -(p * np.log2(p + 1e-12)).sum()
            # 方差（近似用 GLCM 概率矩）
            i = np.arange(levels)
            j = np.arange(levels)
            I, J = np.meshgrid(i, j, indexing='ij')
            mu = (I * p.sum(axis=(2,3))).sum()
            var = (((I - mu)**2) * p.sum(axis=(2,3))).sum()

            feats[y, x, :] = (var, asm, dissim, contrast, homo, entropy)

    if mask is not None:
        for k in range(6):
            tmp = feats[..., k]
            tmp[mask == 0] = np.nan
            feats[..., k] = tmp
    return feats

def objects_from_binary(bin_mask, min_area=50, closing=3):
    """ 二值掩膜 -> 闭运算 -> 连通域 -> 过滤小斑块，返回标号图 """
    if closing > 1:
        se = morphology.square(closing)
        bin_mask = morphology.binary_closing(bin_mask, selem=se)
    lab = measure.label(bin_mask.astype(np.uint8), connectivity=2)
    keep = np.zeros_like(lab)
    idx = 1
    for region in measure.regionprops(lab):
        if region.area >= min_area:
            keep[lab == region.label] = idx
            idx += 1
    return keep

def metrics_from_cm(cm):
    """ cm: [[TN,FP],[FN,TP]]；返回 PA/UA/OA/F1（阳性类=异常=1） """
    TN, FP, FN, TP = cm.ravel()
    eps = 1e-12
    PA = TP / (TP + FN + eps)            # recall / producer's accuracy
    UA = TP / (TP + FP + eps)            # precision / user's accuracy
    OA = (TP + TN) / (TP + TN + FP + FN + eps)
    F1 = 2 * UA * PA / (UA + PA + eps)   # PDF 式(2)
    return dict(PA=PA, UA=UA, OA=OA, F1=F1)

# =========================
# 主流程
# =========================
# 读取影像与真值
img = io.imread(image_path)
gray = to_gray(img)
H, W = gray.shape

gt = io.imread(gt_mask_path)
if gt.ndim == 3:
    gt = color.rgb2gray(gt)
gt = (gt > 0.5).astype(np.uint8)  # 1=异常, 0=正常

if impervious_mask_path:
    imp = io.imread(impervious_mask_path)
    if imp.ndim == 3:
        imp = color.rgb2gray(imp)
    imp = (imp > 0.5).astype(np.uint8)
else:
    # 若无人工表面掩膜，则默认全图（实际应用建议先限定到人工表面区域）
    imp = np.ones_like(gt, dtype=np.uint8)

# 1) 在人工表面范围内生成对象（连通域）
obj_labels = objects_from_binary(imp.astype(bool), min_area=min_object_area, closing=closing_size)
num_objs = obj_labels.max()
print(f"[INFO] 对象数量: {num_objs}")

# 2) 计算像素级 GLCM 六特征（PDF 指定的六个特征）
img_u8 = (norm01(gray) * 255).astype(np.uint8)
feats = glcm_props(img_u8, mask=imp, win=7, levels=256)

# 3) 将像素特征聚合到对象（取对象内均值）；得到对象级真值（多数投票）
VAR, ASM, DISS, CONT, HOMO, ENTR = 0,1,2,3,4,5
obj_feat = np.zeros((num_objs, 6), np.float32)
obj_gt   = np.zeros((num_objs,), np.uint8)

for oid in range(1, num_objs+1):
    mask = (obj_labels == oid)
    if mask.sum() == 0:
        continue
    obj_feat[oid-1, :] = np.nanmean(feats[mask], axis=0)
    obj_gt[oid-1] = 1 if (gt[mask].mean() >= 0.5) else 0

# 4) ASAI（PDF 式(1)）：(Entropy - Homogeneity) / (Entropy + Homogeneity)
HOMO_n = norm01(obj_feat[:, HOMO])
ENTR_n = norm01(obj_feat[:, ENTR])
asai = (ENTR_n - HOMO_n) / (ENTR_n + HOMO_n + 1e-12)
asai_pred = (asai > asai_threshold).astype(np.uint8)

# 5) 随机森林（使用同一组 GLCM 特征做对象级分类）
X = obj_feat.copy()
y = obj_gt.copy()

# 推荐留出测试集做公平对比
if (y == 1).sum() > 1 and (y == 0).sum() > 1:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    # 为了可比性，也把 ASAI 在同一测试集上评估
    asai_te = asai[y.index if hasattr(y, "index") else slice(None)] if hasattr(y, "__array_interface__") else asai
    asai_pred_te = asai_pred[y_tr.shape[0]:] if len(asai_pred) == len(y) else asai_pred  # 简化处理
    # 更稳妥：根据测试索引直接取
    _, X_te_idx, _, y_te_idx = train_test_split(np.arange(len(X)), y, test_size=0.3, stratify=y, random_state=42)
    asai_pred_te = asai_pred[y_te_idx]
    cm_asai = confusion_matrix(y[y_te_idx], asai_pred_te, labels=[0,1])
    cm_rf   = confusion_matrix(y_te, rf_pred, labels=[0,1])
else:
    # 少量样本情形，退化为“全样本回代”评估（注意会偏乐观）
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X, y)
    rf_pred = rf.predict(X)
    cm_asai = confusion_matrix(y, asai_pred, labels=[0,1])
    cm_rf   = confusion_matrix(y, rf_pred,   labels=[0,1])

m_asai = metrics_from_cm(cm_asai)
m_rf   = metrics_from_cm(cm_rf)

print("\n=== ASAI 结果（对象级） ===")
print(f"PA(recall): {m_asai['PA']:.4f}, UA(precision): {m_asai['UA']:.4f}, OA: {m_asai['OA']:.4f}, F1: {m_asai['F1']:.4f}")
print("混淆矩阵 [[TN,FP],[FN,TP]]:\n", cm_asai)

print("\n=== RF 结果（对象级） ===")
print(f"PA(recall): {m_rf['PA']:.4f}, UA(precision): {m_rf['UA']:.4f}, OA: {m_rf['OA']:.4f}, F1: {m_rf['F1']:.4f}")
print("混淆矩阵 [[TN,FP],[FN,TP]]:\n", cm_rf)

# 6) （可选）将对象矢量化并写出方法结果属性，便于GIS查看
outfile = "rf_asai_objects.shp"
geoms, ids, rf_cls, asai_cls, asai_val = [], [], [], [], []
for region in measure.regionprops(obj_labels):
    mask = (obj_labels == region.label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    # 取最大轮廓
    c = max(contours, key=cv2.contourArea)
    if len(c) >= 3:
        poly = Polygon([tuple(pt[0][::-1]) for pt in c])  # (col,row)->(x,y)
        if poly.is_valid and poly.area > 0:
            idx = region.label - 1
            geoms.append(poly)
            ids.append(region.label)
            rf_cls.append(int(rf.predict(obj_feat[idx:idx+1])[0]))
            asai_cls.append(int(asai_pred[idx]))
            asai_val.append(float(asai[idx]))

if geoms:
    gdf = gpd.GeoDataFrame(
        {"obj_id": ids, "rf_cls": rf_cls, "asai_cls": asai_cls, "asai_val": asai_val},
        geometry=geoms, crs="EPSG:3857"
    )
    gdf.to_file(outfile)
    print(f"\n[INFO] 对象矢量已输出: {outfile}")
