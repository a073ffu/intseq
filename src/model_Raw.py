import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import time
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- ハイパーパラメータの定義 ---
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_TEST_SPLIT_RATIO = 0.8 
DROPOUT = 0.4

PRIMES=(2,3,5,7)
MAX_VP=40

N_BINS = 20

# --- 分析に使用するサンプル数 ---
MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION = 50000

# ===== 共通ユーティリティ =====
def l2n(x):  # ベクトルL2正規化
    return tf.nn.l2_normalize(x, axis=-1)

def cosine_distance(a, b):
    a = l2n(a); b = l2n(b)
    return 1.0 - tf.reduce_sum(a * b, axis=-1, keepdims=True)  # [0,2]

def margin_ranking_loss(y_pm1, dAB, dAC, margin=0.2):
    # y ∈ {+1, -1}（+1:「Bの方が近い」）を想定
    return tf.reduce_mean(tf.maximum(0.0, margin - y_pm1 * (dAC - dAB)))

# ===== 各ストリームの極小エンコーダ =====
def make_seq_encoder(input_shape, emb_dim=128, dropout=DROPOUT, name="enc_seq"):
    """連続特徴（float32, shape=(L, F)）用: Conv1D→GAP→Dense→L2"""
    inp = keras.Input(shape=input_shape, dtype="float32", name=f"{name}_in")
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(emb_dim, activation="relu")(x)
    x = layers.Lambda(l2n)(x)
    return keras.Model(inp, x, name=name)

def make_factor_encoder(seq_len, chan_dim,  # chan_dim = 2 * len(primes)  (mod p, v_p)
                        mod_vocabs, vp_vocab, token_dim=8, emb_dim=128,
                        dropout=DROPOUT, name="enc_factor"):
    """mod/vp（int32, shape=(L, chan_dim)）用: primeごとにEmbedding→結合→Conv1D→GAP→Dense→L2"""
    inp = keras.Input(shape=(seq_len, chan_dim), dtype="int32", name=f"{name}_in")
    embedded = []
    num_primes = chan_dim // 2
    for i in range(num_primes):
        mod_i = layers.Lambda(lambda t, idx=i: t[:, :, 2*idx])(inp)      # (B,L)
        vp_i  = layers.Lambda(lambda t, idx=i: t[:, :, 2*idx+1])(inp)    # (B,L)
        mod_e = layers.Embedding(input_dim=mod_vocabs[i], output_dim=token_dim)(mod_i)
        vp_e  = layers.Embedding(input_dim=vp_vocab,      output_dim=token_dim)(vp_i)
        embedded.append(layers.Concatenate(axis=-1)([mod_e, vp_e]))      # (B,L, 2*token_dim)
    x = layers.Concatenate(axis=-1)(embedded)  # (B,L, num_primes*2*token_dim)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(emb_dim, activation="relu")(x)
    x = layers.Lambda(l2n)(x)
    return keras.Model(inp, x, name=name)

def make_bin_encoder(seq_len, n_bins, token_dim=16, emb_dim=128, dropout=DROPOUT, name="enc_bin"):
    """分位ビン列（int32, shape=(L,)）用: Embedding→Conv1D→GAP→Dense→L2"""
    inp = keras.Input(shape=(seq_len,), dtype="int32", name=f"{name}_in")
    x = layers.Embedding(input_dim=n_bins, output_dim=token_dim, mask_zero=False)(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(emb_dim, activation="relu")(x)
    x = layers.Lambda(l2n)(x)
    return keras.Model(inp, x, name=name)

# ===== ゲート付き合併レイヤ（学習可能スカラー3つ） =====
class GatedConcatL2(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shapes):
        # 3ストリーム前提（Seq, Factor, Bin）
        self.alpha_seq    = self.add_weight("alpha_seq",    shape=(), initializer="zeros", trainable=True, dtype=tf.float32)
    def call(self, inputs):
        z_seq = inputs
        g_seq    = tf.nn.softplus(self.alpha_seq)
        z = tf.concat([g_seq*z_seq], axis=-1)
        return l2n(z)

# ===== モデル構築（Siamese：A/B/Cで重み共有） =====
def build_siamese_seq_factor(
    shape_seq,          # 例: (MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)        float32
    emb_dim=128,
    margin=0.2,
    lr=1e-3
):
    # 共有エンコーダ
    enc_seq    = make_seq_encoder(shape_seq, emb_dim=emb_dim, name="enc_seq")
    shared_gated_concat = GatedConcatL2(name="shared_gated_concat_l2")

    # 入力（数列A/B/C × 3ストリーム）
    A_seq = keras.Input(shape=shape_seq,   dtype="float32", name="A_seq")

    B_seq = keras.Input(shape=shape_seq,   dtype="float32", name="B_seq")

    C_seq = keras.Input(shape=shape_seq,   dtype="float32", name="C_seq")

    # 各ストリームをエンコード
    def embed_triplet(inp_seq):
        z_seq    = enc_seq(inp_seq)
        z = shared_gated_concat([z_seq])
        return z

    zA = embed_triplet(A_seq)
    zB = embed_triplet(B_seq)
    zC = embed_triplet(C_seq)

    # 距離
    dAB = layers.Lambda(lambda t: cosine_distance(t[0], t[1]), name="dAB")([zA, zB])  # (B,1)
    dAC = layers.Lambda(lambda t: cosine_distance(t[0], t[1]), name="dAC")([zA, zC])  # (B,1)

    # ラベル (+1 / -1) を別入力で受けて add_loss
    y_pm1 = keras.Input(shape=(1,), dtype="float32", name="y_pm1")
    loss  = layers.Lambda(lambda t: margin_ranking_loss(t[0], t[1], t[2], margin))([y_pm1, dAB, dAC])

    model = keras.Model(
        inputs=[A_seq, B_seq, C_seq, y_pm1],
        outputs=[dAB, dAC],
        name="siamese_seq_factor_bin"
    )
    model.add_loss(loss)
    model.compile(optimizer=keras.optimizers.Adam(lr))
    return model

# --- データ前処理 ---
def signed_log1p(x: np.ndarray) -> np.ndarray:
    # 符号付き対数
    x = x.astype(np.float64, copy=False)
    return np.sign(x) * np.log1p(np.abs(x))

def quantile_clip(x, q=0.995):
    # ロバスト剪断
    hi = np.quantile(np.abs(x), q)
    if hi > 0: x = np.clip(x, -hi, hi)
    return x

def per_sequence_scale(feat_2d: np.ndarray):
    # feat_2d shape: (L, F) ここでは主に raw or signed_log の1列を基準にしてOK
    s = np.median(np.abs(feat_2d[:, 0]))  # 代表チャネルで決める（他でも可）
    s = max(1.0, s)
    scaled = feat_2d / s
    log_s_col = np.full((feat_2d.shape[0], 1), np.log(s + 1e-8), dtype=np.float64)
    return np.concatenate([scaled, log_s_col], axis=1)

# --- 数列を符号付対数，1次差分，窓統計の形に変換 ---
def make_seq_features_intsafe(seq, target_len=20, use_clip=True, q=0.995):
    # 基本列
    raw = np.asarray(seq, dtype=np.float64)
    L = len(raw)

    # 符号付き対数（主）
    log_sig = signed_log1p(raw)
    if use_clip:
        log_sig = quantile_clip(log_sig, q)

    # 1次差分（整数→浮動小数直前に整形）
    diff1 = np.diff(raw, prepend=raw[0]).astype(np.float64)
    diff1 = signed_log1p(diff1)  # 差分も対数圧縮
    if use_clip:
        diff1 = quantile_clip(diff1, q)

    # 窓統計は log_sig 上で
    def window_mean_std(x, k):
        half = k//2
        means, stds = [], []
        for i in range(L):
            s, e = max(0, i-half), min(L, i+half+1)
            w = x[s:e]
            if len(w) < k:
                pad_left = (k - len(w))//2
                pad_right = k - len(w) - pad_left
                w = np.pad(w, (pad_left, pad_right), constant_values=0.0)
            means.append(np.mean(w))
            stds.append(np.std(w))
        return np.column_stack([means, stds])

    win_k3 = window_mean_std(log_sig, k=3) # 窓３

    #win_k5 = window_mean_std(log_sig, k=5) # 窓５

    # 2D特徴 [log_sig, diff1, win_mean, win_std]
    feats = np.column_stack([log_sig, diff1, win_k3
                             #, win_k5
                             ])

    # per-seq scaling + スケール情報付与
    feats = per_sequence_scale(feats)  # 列末に log(s) が追加される

    # pad/trunc
    if L < target_len:
        pad = target_len - L
        feats = np.pad(feats, ((0, pad), (0, 0)), constant_values=0.0)
    else:
        feats = feats[:target_len, :]

    return feats.astype(np.float32)  # 学習直前にfloat32

SEQ_INPUT_NUM = 1

# --- 数列を素数剰余とp進指数形に変換 ---
def vp(x, p, max_vp=40):
    """p進指数 v_p(x) を計算 (xが0のとき0扱い, 最大値はクリップ)"""
    if x == 0:
        return 0
    k = 0
    while x % p == 0:
        x //= p
        k += 1
        if k >= max_vp:
            return max_vp
    return k

def transform_sequence_mod_vp(seq, primes=(2, 3, 5, 7), max_vp=40):
    """
    数列 seq (list[int]) を [mod p, v_p(x)] 特徴に変換
    出力 shape = (len(seq), len(primes)*2)
    """
    features = []
    for x in seq:
        feats_x = []
        for p in primes:
            feats_x.append(int(x % p))         # mod p 値
            feats_x.append(vp(abs(x), p, max_vp))  # v_p(x) (絶対値で計算)
        features.append(feats_x)
    return np.array(features, dtype=np.int32)

# --- 差分数列分位ビン化用関数 ---
# Δ化関数
def transform_sequence_diff(seq):
    seq = np.array(seq, dtype=np.float64)
    return np.diff(seq)

def compute_bin_edges(all_sequences, n_bins=20):
    # 全数列を1Dに平坦化し，ビン範囲を決定
    # 出力は境界点のリスト n_bins+1
    all_values = np.concatenate(all_sequences)
    edges = np.quantile(all_values, np.linspace(0, 1, n_bins+1))
    return edges

def sequence_to_bins(seq, edges):
    # ビン化
    bins = np.searchsorted(edges, seq, side='right') - 1
    return np.clip(bins, 0, len(edges)-2)  # IDは0〜n_bins-1

# --- メイン処理 ---
if __name__ == "__main__":
    
    # ★ 1. ロードするデータセットファイルの指定 ★
    DATASET_FILES = [
        "comparison_datasets/comparison_data_ver3_fixed_A.pkl"# generate_comparison_datasets.py で生成したファイル名に合わせる
        # 例: 複数の深さの比較データを結合する場合
        # "comparison_datasets/comparison_data_depth_1-10.pkl",
        # "comparison_datasets/comparison_data_depth_11-15.pkl",
    ]

    # --- 2. データセットのロードと結合 ---
    all_raw_samples = [] 

    print("\n--- Loading datasets from specified files ---")
    for file_path in DATASET_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue
        
        start_time_load = time.time()
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            end_time_load = time.time()
            print(f"Loaded {len(data)} samples from {file_path} in {end_time_load - start_time_load:.2f} seconds.")
            all_raw_samples.extend(data) 
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this file.")
            continue
    
    if not all_raw_samples:
        print("Error: No data loaded from any specified files. Please check DATASET_FILES paths.")
        exit()

    print(f"\nTotal raw samples loaded: {len(all_raw_samples)}")

    # --- 3. 必要サンプル数のチェックとサンプリング ---
    if len(all_raw_samples) < MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION:
        print(f"ERROR: Total loaded samples ({len(all_raw_samples)}) is less than required ({MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Aborting.")
        exit()
    elif len(all_raw_samples) > MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION:
        print(f"Total loaded samples ({len(all_raw_samples)}) exceeds required ({MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Sampling randomly...")
        
        indices = np.arange(len(all_raw_samples))
        np.random.shuffle(indices)
        sampled_indices = indices[:MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION]
        sampled_samples_raw = [all_raw_samples[i] for i in sampled_indices]
        
        print(f"Sampled down to {len(sampled_samples_raw)} samples for training and evaluation.")
    else:
        sampled_samples_raw = all_raw_samples
        print(f"Using all {len(sampled_samples_raw)} loaded samples for training and evaluation.")
    
    # --- 4. データ準備 (3つの入力) ---
    X_A_raw = np.array([s['numeric_sequence_A'] for s in sampled_samples_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH)
    X_B_raw = np.array([s['numeric_sequence_B'] for s in sampled_samples_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH)
    X_C_raw = np.array([s['numeric_sequence_C'] for s in sampled_samples_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH)
    y_labels_raw = [s['target_label'] for s in sampled_samples_raw] 

    # NumPy変換と数列の変換
    X_A_seq = np.array([signed_log1p(seq) for seq in X_A_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    X_B_seq = np.array([signed_log1p(seq) for seq in X_B_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    X_C_seq = np.array([signed_log1p(seq) for seq in X_C_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    y_labels = np.array(y_labels_raw, dtype=np.int32)

    # 訓練データとテストデータに分割
    # 3つの入力 (X_A_seq, X_B_seq, X_C_seq) と1つのターゲット (y_labels)
    X_train_A_seq, X_test_A_seq, X_train_B_seq, X_test_B_seq, X_train_C_seq, X_test_C_seq, \
    y_train, y_test = train_test_split(
        X_A_seq, X_B_seq, X_C_seq, y_labels,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO, 
        random_state=42,
        stratify=y_labels # 分類問題なのでstratifyでラベル分布を維持
    )

    # 訓練データを検証データと分割
    X_train_A_seq, X_val_A_seq, X_train_B_seq, X_val_B_seq, X_train_C_seq, X_val_C_seq, \
    y_train, y_val = train_test_split(
        X_train_A_seq, X_train_B_seq, X_train_C_seq, y_train,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO, 
        random_state=42,
        stratify=y_train # 分類問題なのでstratifyでラベル分布を維持
    )


    # 形状（あなたの配列から取得）
    shape_seq    = X_train_A_seq.shape[1:]       # (L, F)

    # 0/1 → +1/-1 に変換
    y_train_pm1 = np.where(y_train == 1, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    y_val_pm1   = np.where(y_val   == 1, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    y_test_pm1  = np.where(y_test  == 1, 1.0, -1.0).astype(np.float32).reshape(-1, 1)

    model = build_siamese_seq_factor(
        shape_seq=shape_seq,
        emb_dim=128,
        margin=0.2,
        lr=1e-3
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    history = model.fit(
        {
            "A_seq": X_train_A_seq, 
            "B_seq": X_train_B_seq,  
            "C_seq": X_train_C_seq,  
            "y_pm1": y_train_pm1,
        },
        epochs=30, batch_size=128,
        validation_data=(
            {
                "A_seq": X_val_A_seq, 
                "B_seq": X_val_B_seq, 
                "C_seq": X_val_C_seq, 
                "y_pm1": y_val_pm1,
            }, None
        ),
        callbacks=callbacks_list,
        verbose=1
    )

    # --- 推論＆評価（dAB < dAC → 予測=+1） ---
    dAB_test, dAC_test = model.predict({
        "A_seq": X_test_A_seq, 
        "B_seq": X_test_B_seq, 
        "C_seq": X_test_C_seq, 
        "y_pm1": y_test_pm1,  # 未使用だが入力は必要
    }, verbose=0)
    y_pred_pm1 = tf.cast((dAB_test.values < dAC_test.values), dtype=tf.int32).numpy()  # True→1, False→0
    y_true_pm1 = (y_test_pm1.flatten() == 1.0).astype(np.int32)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    print("ACC:", accuracy_score(y_true_pm1, y_pred_pm1))
    print("P  :", precision_score(y_true_pm1, y_pred_pm1))
    print("R  :", recall_score(y_true_pm1, y_pred_pm1))
    print("F1 :", f1_score(y_true_pm1, y_pred_pm1))
    # 参考で AUC（距離差 dAC - dAB をスコアに）
    scores = (dAC_test.values - dAB_test.values).numpy().flatten()
    print("AUC:", roc_auc_score(y_true_pm1, scores))
# ランダム数列A
# ★ 1. ロードするデータセットファイルの指定 ★
    DATASET_FILES = [
        "comparison_datasets/comparison_data_ver3_randam_A.pkl"# generate_comparison_datasets.py で生成したファイル名に合わせる
        # 例: 複数の深さの比較データを結合する場合
        # "comparison_datasets/comparison_data_depth_1-10.pkl",
        # "comparison_datasets/comparison_data_depth_11-15.pkl",
    ]

    # --- 2. データセットのロードと結合 ---
    all_raw_samples = [] 

    print("\n--- Loading datasets from specified files ---")
    for file_path in DATASET_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue
        
        start_time_load = time.time()
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            end_time_load = time.time()
            print(f"Loaded {len(data)} samples from {file_path} in {end_time_load - start_time_load:.2f} seconds.")
            all_raw_samples.extend(data) 
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this file.")
            continue
    
    if not all_raw_samples:
        print("Error: No data loaded from any specified files. Please check DATASET_FILES paths.")
        exit()

    print(f"\nTotal raw samples loaded: {len(all_raw_samples)}")

    # --- 3. 必要サンプル数のチェックとサンプリング ---
    if len(all_raw_samples) < 10000:
        print(f"ERROR: Total loaded samples ({len(all_raw_samples)}) is less than required ({MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Aborting.")
        exit()
    elif len(all_raw_samples) > 10000:
        print(f"Total loaded samples ({len(all_raw_samples)}) exceeds required ({MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Sampling randomly...")
        
        indices = np.arange(len(all_raw_samples))
        np.random.shuffle(indices)
        sampled_indices = indices[:10000]
        sampled_samples_raw = [all_raw_samples[i] for i in sampled_indices]
        
        print(f"Sampled down to {len(sampled_samples_raw)} samples for training and evaluation.")
    else:
        sampled_samples_raw = all_raw_samples
        print(f"Using all {len(sampled_samples_raw)} loaded samples for training and evaluation.")
    
    # --- 4. データ準備 (3つの入力) ---
    rX_A_raw = np.array([s['numeric_sequence_A'] for s in sampled_samples_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH)
    rX_B_raw = np.array([s['numeric_sequence_B'] for s in sampled_samples_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH)
    rX_C_raw = np.array([s['numeric_sequence_C'] for s in sampled_samples_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH)
    ry_labels_raw = [s['target_label'] for s in sampled_samples_raw] 

    # NumPy変換と数列の変換
    rX_A_seq = np.array([signed_log1p(seq) for seq in rX_A_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    rX_B_seq = np.array([signed_log1p(seq) for seq in rX_B_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    rX_C_seq = np.array([signed_log1p(seq) for seq in rX_C_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    ry_labels = np.array(ry_labels_raw, dtype=np.int32)

    ry_test_pm1  = np.where(ry_labels  == 1, 1.0, -1.0).astype(np.float32).reshape(-1, 1)

    # 訓練データとテストデータに分割
    # 3つの入力 (X_A_seq, X_B_seq, X_C_seq) と1つのターゲット (y_labels)

    print("--- Check rX Types and Shapes ---")
    # 全ての rX_* 変数について確認
    for name, var in {
        "rX_A_seq": rX_A_seq,  
        "rX_B_seq": rX_B_seq,  
        "rX_C_seq": rX_C_seq,  
        "ry_test_pm1": ry_test_pm1,
    }.items():
        print(f"Variable: {name}")
        print(f"  Type: {type(var)}")
        if hasattr(var, 'shape'):
            print(f"  Shape: {var.shape}")
        else:
            # このメッセージが出た変数こそがエラーの原因です
            print(f"  --- ERROR: Has no shape attribute (Likely a String or None) ---")

    # --- 推論＆評価（dAB < dAC → 予測=+1） ---
    rdAB_test, rdAC_test = model.predict({
        "A_seq": rX_A_seq,  
        "B_seq": rX_B_seq,  
        "C_seq": rX_C_seq, 
        "y_pm1": ry_test_pm1,  # 未使用だが入力は必要
    }, verbose=0)
    ry_pred_pm1 = tf.cast((rdAB_test.values < rdAC_test.values), dtype=tf.int32).numpy()  # True→1, False→0
    ry_true_pm1 = (ry_test_pm1.flatten() == 1.0).astype(np.int32)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    print("ACC:", accuracy_score(ry_true_pm1, ry_pred_pm1))
    print("P  :", precision_score(ry_true_pm1, ry_pred_pm1))
    print("R  :", recall_score(ry_true_pm1, ry_pred_pm1))
    print("F1 :", f1_score(ry_true_pm1, ry_pred_pm1))
    # 参考で AUC（距離差 dAC - dAB をスコアに）
    rscores = (rdAC_test.values - rdAB_test.values).numpy().flatten()
    print("AUC:", roc_auc_score(ry_true_pm1, rscores))