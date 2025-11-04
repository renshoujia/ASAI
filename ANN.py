import numpy as np
import cv2
from skimage import io, color, measure, morphology, feature
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from shapely.geometry import Polygon
import geopandas as gpd

# =========================

# =========================
image_path = "your_image.png"          # 原始影像路径（灰度或RGB）
gt_mask_path = "your_gt_mask.png"      # 二值GT：1=异常人工表面, 0=正常人工表面
impervious_mask_path = None            # 可选：人工表面区域掩膜(1=人工表面)，若无则设为None

# 对象生成参数
min_object_area = 50                   # 连通域最小像素数（去噪）
closing_size = 3                       # 形态学闭运算 kernel size（对象更连贯）

# ASAI 判别阈值（PDF 式(1)为差/和，数值范围[-1,1]；经验上>0 表示Entropy>Homogeneity）
asai_threshold = 0.0

# ANN 参数（与PDF中“MLP”一致的思路:contentReference[oaicite:3]{index=3}）
mlp_params = dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                  max_iter=300, random_state=42)

# =========================
# 工具函数
# =========================
def to_gray(img):
    if img.ndim == 3:
        return color.rgb2gray(img)
    return img.astype(np.float32) / (img.max() if img.max() > 1 else 1.0)

def norm01(arr, eps=1e-12):
    m, M = np.nanmin(arr), np.nanmax(arr)
    if M - m < eps:
        return np.zeros_like(arr)
    return (arr - m) / (M - m)

def glcm_props(img_u8, mask=None, win=7, levels=256, distances=[1], angles=[0]):
    """
    基于滑窗的 GLCM 特征（六个）：Variance, SecondMoment, Dissimilarity, Contrast, Homogeneity, Entropy
    返回 shape = (H, W, 6)
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
            # 常见属性
            contrast = feature.greycoprops(glcm, 'contrast').mean()
            dissim  = feature.greycoprops(glcm, 'dissimilarity').mean()
            homo    = feature.greycoprops(glcm, 'homogeneity').mean()
            asm     = feature.greycoprops(glcm, 'ASM').mean()  # Angular Second Moment
            # 熵（自定义）
            p = glcm.astype(np.float64)
            entropy = -(p * np.log2(p + 1e-12)).sum()

            # 方差（基于 GLCM 概率矩计算）
            i = np.arange(levels)
            j = np.arange(levels)
            I, J = np.meshgrid(i, j, indexing='ij')
            mu = (I * p.sum(axis=(2,3))).sum()  # 简化：平均灰度（近似）
            var = (((I - mu)**2) * p.sum(axis=(2,3))).sum()

            feats[y, x, :] = (var, asm, dissim, contrast, homo, entropy)

    if mask is not None:
        for k in range(feats.shape[-1]):
            tmp = feats[..., k]
            tmp[mask == 0] = np.nan
            feats[..., k] = tmp
    return feats  # (H,W,6)

def objects_from_binary(bin_mask, min_area=50, closing=3):
    """ 二值掩膜 -> 形态学闭运算 -> 连通组件 -> 过滤小斑块 """
    if closing > 1:
        se = morphology.square(closing)
        bin_mask = morphology.binary_closing(bin_mask, selem=se)
    lab = measure.label(bin_mask.astype(np.uint8), connectivity=2)
    # 区域过滤
    keep = np.zeros_like(lab)
    idx = 1
    for region in measure.regionprops(lab):
        if region.area >= min_area:
            keep[lab == region.label] = idx
            idx += 1
    return keep  # 连通域标号图：0=背景, 1..N

def vectorize_objects(label_img, outfile="objects.shp"):
    geoms, ids = [], []
    for region in measure.regionprops(label_img):
        # 使用 region 边界栅格化等高线 -> 多边形
        mask = (label_img == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) >= 3:
                poly = Polygon([tuple(pt[0][::-1]) for pt in c])  # (col,row)->(x,y)
                if poly.is_valid and poly.area > 0:
                    geoms.append(poly)
                    ids.append(region.label)
    if geoms:
        gdf = gpd.GeoDataFrame({'obj_id': ids}, geometry=geoms, crs="EPSG:3857")
        gdf.to_file(outfile)
    return outfile

def metrics_from_cm(cm):
    """
    cm 顺序：[[TN, FP],[FN, TP]]，以类别1=“异常”为阳性类
    PA(recall)=TP/(TP+FN)；UA(precision)=TP/(TP+FP)；OA=(TP+TN)/N；F1=2*UA*PA/(UA+PA)
    """
    TN, FP, FN, TP = cm.ravel()
    eps = 1e-12
    PA = TP / (TP + FN + eps)
    UA = TP / (TP + FP + eps)
    OA = (TP + TN) / (TP + TN + FP + FN + eps)
    F1 = 2 * UA * PA / (UA + PA + eps)
    return dict(PA=PA, UA=UA, OA=OA, F1=F1)

# =========================
# 主流程
# =========================
# 读入数据
img = io.imread(image_path)
gray = to_gray(img)
H, W = gray.shape

gt = io.imread(gt_mask_path)
if gt.ndim == 3:
    gt = color.rgb2gray(gt)
gt = (gt > 0.5).astype(np.uint8)   # 1=异常, 0=正常

if impervious_mask_path:
    imp = io.imread(impervious_mask_path)
    if imp.ndim == 3:
        imp = color.rgb2gray(imp)
    imp = (imp > 0.5).astype(np.uint8)
else:
    # 若未提供人工表面掩膜，则在整幅图上形成对象（实际应用建议先限定为人工表面区域）
    imp = np.ones_like(gt, dtype=np.uint8)

# 1) 生成对象（在人造面范围内）
obj_labels = objects_from_binary(imp.astype(bool), min_area=min_object_area, closing=closing_size)
num_objs = obj_labels.max()
print(f"[INFO] 对象数量: {num_objs}")

# 2) 计算 GLCM 六特征（PDF 第9–10页：Variance, Second Moment(ASM), Dissimilarity, Contrast, Homogeneity, Entropy）
img_u8 = (norm01(gray) * 255).astype(np.uint8)
feats = glcm_props(img_u8, mask=imp, win=7, levels=256, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])

# 3) 将像素级特征汇聚到对象：对每个对象取均值
obj_feat = np.zeros((num_objs, 6), np.float32)
obj_gt   = np.zeros((num_objs,), np.uint8)  # 1=异常,0=正常（对象内多数投票）
for oid in range(1, num_objs+1):
    mask = (obj_labels == oid)
    if mask.sum() == 0:
        continue
    # 对象均值特征
    vals = feats[mask]
    obj_feat[oid-1, :] = np.nanmean(vals, axis=0)

    # 对象真值（多数投票）
    gtv = gt[mask]
    obj_gt[oid-1] = 1 if (gtv.mean() >= 0.5) else 0

# 拆解命名
VAR, ASM, DISS, CONT, HOMO, ENTR = 0,1,2,3,4,5

# 4) ASAI 判别（PDF 第10页 式(1)）
# 归一化到[0,1]（与文中“归一化到0-1后再入式(1)”一致），然后计算 (Entropy - Homogeneity)/(Entropy + Homogeneity)
HOMO_n = norm01(obj_feat[:, HOMO])
ENTR_n = norm01(obj_feat[:, ENTR])
asai = (ENTR_n - HOMO_n) / (ENTR_n + HOMO_n + 1e-12)
asai_pred = (asai > asai_threshold).astype(np.uint8)  # >0 表示“熵主导”，经验上偏异常

# 5) ANN（MLP）分类（PDF 第14–16页：与 ASAI 对比的 ANN/MLP）
# 文中“同一组纹理特征”思路，这里用六个 GLCM 特征
X = obj_feat.copy()
y = obj_gt.copy()

# 简单划分：用全部对象做训练并回代评估（若有样本足够，建议留出测试集/交叉验证）
clf = MLPClassifier(**mlp_params)
clf.fit(X, y)
ann_pred = clf.predict(X)

# 6) 评估（PDF 第11页 式(2)：F1 = 2*UA*PA/(UA+PA)，并报告 PA/UA/OA）
cm_asai = confusion_matrix(y
