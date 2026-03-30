import os
import cv2
import numpy as np
import glob
import time
import random
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import (
    precision_score,
    f1_score,
    recall_score, balanced_accuracy_score, classification_report, accuracy_score
)
import pandas as pd
import albumentations as A

# 配置
IMG_SIZE = (128, 128)
FEATURE_CACHE = "features_cache_v4.npz"
USE_FEATURE_SELECTION = True
K_BEST_FEATURES = 500
CALIBRATE_MODELS = False

# 数据增强策略（仅用于训练）
AUGMENTATION = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.9, 1.1),
        rotate=0,
        p=0.2
    )
])


# 特征提取函数

def extract_multiscale_lbp(img):
    feats = []
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(img, P=n_points, R=radius, method="uniform")
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        feats.append(hist)
    return np.concatenate(feats)


def extract_enhanced_fft_features(image):
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log(mag + 1e-8)

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r_max = min(h, w) // 2

    regions = [(0, r_max // 3), (r_max // 3, 2 * r_max // 3), (2 * r_max // 3, r_max)]
    features = []
    for r1, r2 in regions:
        mask = (x - cx) ** 2 + (y - cy) ** 2
        region_mask = (mask >= r1 ** 2) & (mask < r2 ** 2)
        vals = log_mag[region_mask]
        if len(vals) == 0:
            mean_val = var_val = p90 = 0.0
        else:
            mean_val = np.mean(vals)
            var_val = np.var(vals)
            p90 = np.percentile(vals, 90)
        features.extend([mean_val, var_val, p90])

    low_mask = (x - cx) ** 2 + (y - cy) ** 2 < (r_max // 3) ** 2
    high_mask = (x - cx) ** 2 + (y - cy) ** 2 >= (2 * r_max // 3) ** 2
    low_energy = np.mean(log_mag[low_mask]) if np.any(low_mask) else 0.0
    high_energy = np.mean(log_mag[high_mask]) if np.any(high_mask) else 1e-8
    energy_ratio = low_energy / (high_energy + 1e-8)
    features.append(energy_ratio)
    return np.array(features)


def extract_noise_features(img):
    blurred1 = cv2.GaussianBlur(img, (3, 3), 0)
    residue1 = img.astype(np.float32) - blurred1
    blurred2 = cv2.medianBlur(img, 3)
    residue2 = img.astype(np.float32) - blurred2
    feats = []
    for res in [residue1, residue2]:
        abs_res = np.abs(res)
        feats.extend([
            np.mean(res),
            np.std(res),
            np.percentile(abs_res, 95),
            np.sum(abs_res > 3 * np.std(res)) / res.size if res.size > 0 else 0.0
        ])
    return np.array(feats)


def extract_edge_features(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.hypot(sobelx, sobely)
    return np.array([np.mean(edge_mag), np.std(edge_mag), np.percentile(edge_mag, 90)])


def extract_dct_features(gray_img):
    h, w = gray_img.shape
    block_size = 8
    coeffs = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray_img[i:i + block_size, j:j + block_size].astype(np.float32)
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            coeffs.append(dct_block[1:4, 1:4].flatten())
    if not coeffs:
        return np.zeros(9)
    return np.mean(coeffs, axis=0)


def extract_features_from_arrays(gray, bgr):
    lbp = extract_multiscale_lbp(gray)
    fft = extract_enhanced_fft_features(gray)
    noise = extract_noise_features(gray)
    edge = extract_edge_features(gray)

    # Color features from BGR
    img_yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    img_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_feats = []
    for ch in [img_yuv[:, :, 1], img_yuv[:, :, 2], img_hsv[:, :, 1], img_hsv[:, :, 2]]:
        color_feats.extend([
            np.mean(ch), np.std(ch),
            np.percentile(ch, 90),
            np.max(ch) - np.min(ch)
        ])
    color = np.array(color_feats)

    dct_feat = extract_dct_features(gray)
    return np.concatenate([lbp, fft, noise, edge, color, dct_feat])


def extract_single_image_features(img_path, augment=False):
    bgr = cv2.imread(img_path)
    if bgr is None:
        dummy_gray = np.zeros(IMG_SIZE, dtype=np.uint8)
        dummy_bgr = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        feat = extract_features_from_arrays(dummy_gray, dummy_bgr)
        return [feat], [0]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    bgr = cv2.resize(bgr, IMG_SIZE)

    features_list = []
    # 原始样本
    feat_orig = extract_features_from_arrays(gray, bgr)
    features_list.append(feat_orig)

    # 数据增强（仅训练时）
    if augment:
        for _ in range(2):  # 每图生成2个增强样本
            augmented = AUGMENTATION(image=bgr)
            aug_bgr = augmented['image']
            aug_gray = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2GRAY)
            aug_feat = extract_features_from_arrays(aug_gray, aug_bgr)
            features_list.append(aug_feat)

    return features_list, [0] * len(features_list)


def load_image_paths_and_labels(real_dir, fake_dir):
    paths = []
    labels = []
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    for ext in extensions:
        real_list = glob.glob(os.path.join(real_dir, ext))
        fake_list = glob.glob(os.path.join(fake_dir, ext))
        paths.extend(real_list)
        labels.extend([0] * len(real_list))
        paths.extend(fake_list)
        labels.extend([1] * len(fake_list))
    return paths, np.array(labels)


# 主程序
if __name__ == "__main__":

    REAL_DIR = "Train/Real"
    FAKE_DIR = "Train/Fake"
    VAL_REAL = "Validation/Real"
    VAL_FAKE = "Validation/Fake"
    TEST_REAL = "Test/Real"
    TEST_FAKE = "Test/Fake"

    # 加载或生成特征缓存
    if os.path.exists(FEATURE_CACHE):
        print("Loading features from cache...")
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        X_train_feat = data['X_train']
        y_train = data['y_train']
        X_val_feat = data['X_val']
        y_val = data['y_val']
        X_test_feat = data['X_test']
        y_test = data['y_test']
        train_paths = data['train_paths'].tolist()
        val_paths = data['val_paths'].tolist()
        test_paths = data['test_paths'].tolist()
    else:
        print("Feature cache not found. Extracting features with augmentation...")

        # 训练集
        train_paths, y_train_orig = load_image_paths_and_labels(REAL_DIR, FAKE_DIR)
        X_train_feats = []
        y_train_labels = []

        for i, (path, label) in enumerate(zip(train_paths, y_train_orig)):
            if i % 1000 == 0:
                print(f"Processing train {i}/{len(train_paths)}")

            do_aug = True  # 对所有训练样本增强

            feats_list, _ = extract_single_image_features(path, augment=do_aug)
            X_train_feats.extend(feats_list)
            y_train_labels.extend([label] * len(feats_list))

        X_train_feat = np.array(X_train_feats)
        y_train = np.array(y_train_labels)

        # 验证集
        val_paths, y_val = load_image_paths_and_labels(VAL_REAL, VAL_FAKE)
        X_val_feat = np.array([extract_single_image_features(p)[0][0] for p in val_paths])

        # 测试集
        test_paths, y_test = load_image_paths_and_labels(TEST_REAL, TEST_FAKE)
        X_test_feat = np.array([extract_single_image_features(p)[0][0] for p in test_paths])

        # 保存缓存
        np.savez_compressed(
            FEATURE_CACHE,
            X_train=X_train_feat, y_train=y_train,
            X_val=X_val_feat, y_val=y_val,
            X_test=X_test_feat, y_test=y_test,
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths
        )
        print(f"Saved augmented features to {FEATURE_CACHE}")
        print(f"Train samples: {X_train_feat.shape[0]} (original: {len(train_paths)})")

    print(f"\nLoaded features:")
    print(f"  Train: {X_train_feat.shape[0]} samples, {X_train_feat.shape[1]} features")
    print(f"  Val:   {X_val_feat.shape[0]} samples")
    print(f"  Test:  {X_test_feat.shape[0]} samples")

    # 标准化 + 特征选择
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    if USE_FEATURE_SELECTION:
        selector = SelectKBest(score_func=f_classif, k=min(K_BEST_FEATURES, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = selector.transform(X_val_scaled)
        X_test_selected = selector.transform(X_test_scaled)
        print(f"Feature selection: {X_train_scaled.shape[1]} → {X_train_selected.shape[1]} features")
    else:
        X_train_selected, X_val_selected, X_test_selected = X_train_scaled, X_val_scaled, X_test_scaled

    n_feat = X_train_selected.shape[1]
    feature_names = [f"feat_{i}" for i in range(n_feat)]
    X_train_df = pd.DataFrame(X_train_selected, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_selected, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_selected, columns=feature_names)

    # 模型定义
    from sklearn.ensemble import RandomForestClassifier

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            n_jobs=-1),
    }

    if CALIBRATE_MODELS:
        from sklearn.calibration import CalibratedClassifierCV

        calibrated_models = {}
        for name, model in models.items():
            if name != "SVM (Linear)":
                calibrated_models[name] = CalibratedClassifierCV(model, method='isotonic', cv=3)
            else:
                calibrated_models[name] = model
        models = calibrated_models

    results = {}

    # 训练并评估每个模型
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_df, y_train)
        train_time = time.time() - start_time

        # 默认阈值 (0.5) 下的准确率（用于日志）
        y_val_pred = model.predict(X_val_df)
        y_test_pred = model.predict(X_test_df)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"{name} - Val Acc: {val_acc:.4f}, Test Acc: {test_acc :.4f}, Time: {train_time:.1f}s")

        # 存储基础结果
        results[name] = {
            'model': model,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time,
            'test_pred_default': y_test_pred,
            'test_proba': None,
            'final_thresh': 0.5,
            'test_bal_acc': balanced_accuracy_score(y_test, y_test_pred),
            'test_fake_recall': recall_score(y_test, y_test_pred, pos_label=1),
            'report': classification_report(y_test, y_test_pred, target_names=["Real", "Fake"])
        }

        # 阈值优化
        print(f"\nOptimizing threshold on validation set for {name} (target: maximize Macro F1, FakeRecall >= 0.80)...")
        y_val_proba = model.predict_proba(X_val_df)[:, 1]
        results[name]['val_proba'] = y_val_proba

        best_thresh = None
        best_f1 = -1.0
        valid_found = False

        # 第一阶段：优先满足 Fake Recall >= 0.80
        for thresh in np.arange(0.30, 0.61, 0.01):
            y_pred = (y_val_proba >= thresh).astype(int)
            fake_rec = recall_score(y_val, y_pred, pos_label=1)
            if fake_rec >= 0.80:
                f1_macro = f1_score(y_val, y_pred, average='macro')
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_thresh = thresh
                    valid_found = True

        # 第二阶段：若未找到满足条件的阈值，则放宽约束
        if not valid_found:
            best_f1 = -1.0
            for thresh in np.arange(0.10, 0.91, 0.01):
                y_pred = (y_val_proba >= thresh).astype(int)
                f1_macro = f1_score(y_val, y_pred, average='macro')
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_thresh = thresh

        best_thresh = float(best_thresh)
        print(f" Selected threshold for {name}: {best_thresh:.2f}")
        val_fake_recall = recall_score(y_val, (y_val_proba >= best_thresh).astype(int), pos_label=1)
        val_fake_prec = precision_size = precision_score(y_val, (y_val_proba >= best_thresh).astype(int), pos_label=1)
        print(
            f"   Val - FakeRecall: {val_fake_recall:.4f}, FakePrecision: {val_fake_prec:.4f}, Macro F1: {best_f1:.4f}")

        # 用新阈值评估测试集
        y_test_proba = model.predict_proba(X_test_df)[:, 1]
        y_test_pred_tuned = (y_test_proba >= best_thresh).astype(int)

        test_bal_acc = balanced_accuracy_score(y_test, y_test_pred_tuned)

        # 更新结果（覆盖默认 0.5 阈值的结果）
        results[name]['test_proba'] = y_test_proba
        results[name]['final_thresh'] = best_thresh
        results[name]['test_bal_acc'] = balanced_accuracy_score(y_test, y_test_pred_tuned)
        results[name]['test_fake_recall'] = recall_score(y_test, y_test_pred_tuned, pos_label=1)
        results[name]['report'] = classification_report(y_test, y_test_pred_tuned, target_names=["Real", "Fake"])

        print(f"\nEvaluating {name} on TEST set with threshold = {best_thresh:.2f}...")
        print(f"{name} Test Balanced Accuracy: {results[name]['test_bal_acc']:.4f}")
        print(f"{name} Test Fake Recall:       {results[name]['test_fake_recall']:.4f}")