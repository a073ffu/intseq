import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import time
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

# --- ハイパーパラメータの定義 ---
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 128
TRAIN_TEST_SPLIT_RATIO = 0.8 
DROPOUT = 0.3

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
def make_seq_encoder(input_shape, emb_dim=128, dropout=0.3, name="enc_seq"):
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
                        dropout=0.3, name="enc_factor"):
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

def make_bin_encoder(seq_len, n_bins, token_dim=16, emb_dim=128, dropout=0.3, name="enc_bin"):
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
        self.alpha_factor = self.add_weight("alpha_factor", shape=(), initializer="zeros", trainable=True, dtype=tf.float32)
        self.alpha_bin    = self.add_weight("alpha_bin",    shape=(), initializer="zeros", trainable=True, dtype=tf.float32)
    def call(self, inputs):
        z_seq, z_factor, z_bin = inputs
        g_seq    = tf.nn.softplus(self.alpha_seq)
        g_factor = tf.nn.softplus(self.alpha_factor)
        g_bin    = tf.nn.softplus(self.alpha_bin)
        z = tf.concat([g_seq*z_seq, g_factor*z_factor, g_bin*z_bin], axis=-1)
        return l2n(z)

# ===== モデル構築（Siamese：A/B/Cで重み共有） =====
def build_siamese_seq_factor_bin(
    shape_seq,          # 例: (MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)        float32
    shape_factor,       # 例: (MAX_SEQUENCE_LENGTH, len(PRIMES)*2)    int32
    shape_bin,          # 例: (MAX_SEQUENCE_LENGTH-1,)                int32
    n_bins,             # 例: N_BINS
    primes=(2,3,5,7),   # mod の語彙（素数）
    vp_vocab=41,        # v_p の語彙（0..MAX_VP）
    emb_dim=128,
    margin=0.2,
    lr=1e-3
):
    # 共有エンコーダ
    enc_seq    = make_seq_encoder(shape_seq, emb_dim=emb_dim, name="enc_seq")
    enc_factor = make_factor_encoder(seq_len=shape_factor[0], chan_dim=shape_factor[1],
                                     mod_vocabs=primes, vp_vocab=vp_vocab,
                                     token_dim=8, emb_dim=emb_dim, dropout=0.3, name="enc_factor")
    enc_bin    = make_bin_encoder(seq_len=shape_bin[0], n_bins=n_bins,
                                  token_dim=16, emb_dim=emb_dim, dropout=0.3, name="enc_bin")
    shared_gated_concat = GatedConcatL2(name="shared_gated_concat_l2")


    # 入力（数列A/B/C × 3ストリーム）
    A_seq = keras.Input(shape=shape_seq,   dtype="float32", name="A_seq")
    A_factor = keras.Input(shape=shape_factor, dtype="int32", name="A_factor")
    A_bin = keras.Input(shape=shape_bin,   dtype="int32",   name="A_bin")

    # 各ストリームをエンコード
    def embed_triplet(inp_seq, inp_factor, inp_bin):
        z_seq    = enc_seq(inp_seq)
        z_factor = enc_factor(inp_factor)
        z_bin    = enc_bin(inp_bin)
        z = shared_gated_concat([z_seq, z_factor, z_bin])
        return z

    zA = embed_triplet(A_seq, A_factor, A_bin)

    x = layers.Dense(128, activation="relu")(zA)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[A_seq, A_factor, A_bin], outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name='auc')]
    )
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

SEQ_INPUT_NUM = 5

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
    # ---  保存ディレクトリ名の設定 ---
    # ディレクトリ名
    EXPERIMENT_NAME = "All_model_DistanceThreshold_single_encoder_baseline" 

    # 保存先ディレクトリの構築
    # (例: outputs/model_v1_baseline_20251231)
    timestamp = datetime.now().strftime("%Y%m%d")
    save_dir = f"outputs/{EXPERIMENT_NAME}_{timestamp}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # ★ 1. ロードするデータセットファイルの指定 ★
    DATASET_FILES = [
        "comparison_datasets/comparison_data_ver2_fixed_A.pkl"# generate_comparison_datasets.py で生成したファイル名に合わせる
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
    X_B_raw = [s['numeric_sequence_B'] for s in sampled_samples_raw]
    dist_list = np.array([s['edit_distance_AB'] for s in sampled_samples_raw])

    # 中央値を閾値 T に設定
    threshold_T = np.median(dist_list)
    print(f"Computed Threshold (Median): {threshold_T}")

    y_labels_raw = (dist_list > threshold_T)

    # NumPy変換と数列の変換
    
    X_B_seq = np.array([make_seq_features_intsafe(seq) for seq in X_B_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH, SEQ_INPUT_NUM)
    X_B_factor = np.array([transform_sequence_mod_vp(seq, PRIMES) for seq in X_B_raw], dtype=np.int32)
    X_B_bin = np.array([transform_sequence_diff(seq) for seq in X_B_raw], dtype=np.float64).reshape(-1, MAX_SEQUENCE_LENGTH-1)
    y_labels = np.array(y_labels_raw, dtype=np.int32)

    # 訓練データとテストデータに分割
    indices = np.arange(len(y_labels))
    # 3つの入力 (X_A_seq, X_B_seq, X_C_seq) と1つのターゲット (y_labels)
    indices_train, indices_test, X_train_A_seq, X_test_A_seq, X_train_A_factor, X_test_A_factor, X_train_A_bin, X_test_A_bin, \
    y_train, y_test = train_test_split(
        indices,
        X_B_seq, X_B_factor, X_B_bin, y_labels,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO, 
        random_state=42,
        stratify=y_labels # 分類問題なのでstratifyでラベル分布を維持
    )

    # 訓練データを検証データと分割
    X_train_A_seq, X_val_A_seq, X_train_A_factor, X_val_A_factor, X_train_A_bin, X_val_A_bin, \
    y_train, y_val = train_test_split(
        X_train_A_seq, X_train_A_factor, X_train_A_bin, y_train,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO, 
        random_state=42,
        stratify=y_train # 分類問題なのでstratifyでラベル分布を維持
    )

    # ビン化
    edges = compute_bin_edges(list(X_train_A_bin), 
                              n_bins=N_BINS)

    # 訓練データの分位境界を用いて，データを分位ビニング
    X_train_A_bin = np.array([sequence_to_bins(seq, edges) for seq in X_train_A_bin], dtype=np.int32)
    X_val_A_bin = np.array([sequence_to_bins(seq, edges) for seq in X_val_A_bin], dtype=np.int32)
    X_test_A_bin = np.array([sequence_to_bins(seq, edges) for seq in X_test_A_bin], dtype=np.int32)
    

    # 形状（あなたの配列から取得）
    shape_seq    = X_train_A_seq.shape[1:]       # (L, F)
    shape_factor = X_train_A_factor.shape[1:]    # (L, 2*len(PRIMES))
    shape_bin    = X_train_A_bin.shape[1:]       # (L-1,)

    model = build_siamese_seq_factor_bin(
        shape_seq=shape_seq,
        shape_factor=shape_factor,
        shape_bin=shape_bin,
        n_bins=N_BINS,
        primes=PRIMES,          # (2,3,5,7)
        vp_vocab=MAX_VP+1,      # 0..MAX_VP
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
            "A_seq": X_train_A_seq, "A_factor": X_train_A_factor, "A_bin": X_train_A_bin
        },
        y = y_train,
        epochs=30, batch_size=128,
        validation_data=(
            {
                "A_seq": X_val_A_seq, "A_factor": X_val_A_factor, "A_bin": X_val_A_bin
            }, y_val
        ),
        callbacks=callbacks_list,
        verbose=1
    )

    metrics = ['loss', 'accuracy']  # 使用する評価関数を指定

    # --------------------------------------------------
    # 1. Matplotlib 全体設定（rcParams）: フォントサイズの統一
    # --------------------------------------------------
    # 論文に適したフォントサイズを設定
    plt.rcParams['font.size'] = 16       # ベースフォントサイズ
    plt.rcParams['axes.titlesize'] = 16  # グラフのタイトル
    plt.rcParams['axes.labelsize'] = 15  # 軸ラベル (X, Y)
    plt.rcParams['legend.fontsize'] = 14 # 凡例
    plt.rcParams['xtick.labelsize'] = 15 # X軸の目盛り
    plt.rcParams['ytick.labelsize'] = 15 # Y軸の目盛り

    # --------------------------------------------------
    # 2. グラフ描画と保存
    # --------------------------------------------------
    # metrics が ['loss', 'accuracy'] などのリストであると仮定

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # historyからデータを取り出す
        plt_train = history.history[metric]
        plt_test = history.history['val_' + metric]
        
        # プロット
        # 訓練データ: 実線 (linestyle='-')
        plt.plot(plt_train, label='Training', color='C0', linestyle='-')
        # テストデータ: 破線 (linestyle='--') にすることで白黒印刷でも区別可能
        plt.plot(plt_test, label='Validation', color='C1', linestyle='--')
        
        # 軸ラベルとタイトル (視認性向上)
        plt.title(f'Learning Curve for {metric.capitalize()}', pad=15) # タイトル
        plt.xlabel('Epoch')    # X軸ラベル
        plt.ylabel(metric.capitalize()) # Y軸ラベル
        
        # 凡例の表示 (loc='best'で最適な位置に自動配置)
        plt.legend(loc='best')
        
        # X軸の目盛りを整数にする（オプション: 連続値でないことを強調）
        plt.xticks(range(0, len(plt_train), max(1, len(plt_train)//5)))

        # Y軸の範囲をデータの最小値と最大値に合わせて調整（オプション）
        # 曲線がグラフ全体に広がるように調整する
        # 以下の2行は、必要に応じてコメントアウトまたは値を調整してください。
        min_val = min(min(plt_train), min(plt_test)) * 0.95
        max_val = max(max(plt_train), max(plt_test)) * 1.05
        plt.ylim(min_val, max_val)

        # 画像ファイルとして保存
        PLOT_FILENAME = f'learning_curve_{metric}.png'
        # dpi=600で高解像度保存（論文推奨）
        plt.savefig(os.path.join(save_dir, PLOT_FILENAME), dpi=600, bbox_inches='tight') 
        plt.close() 

        print(f"\nLearning curve for {metric} saved to {PLOT_FILENAME}")

    # --- 推論＆評価 ---
    y_pred_prob = model.predict({
        "A_seq": X_test_A_seq, "A_factor": X_test_A_factor, "A_bin": X_test_A_bin, 
        "y_pm1": y_test,  # 未使用だが入力は必要
    }, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(np.int32)
    y_true = (y_test.flatten() == 1.0).astype(np.int32)

    test_samples_meta = [sampled_samples_raw[i] for i in indices_test]

    analysis_df = pd.DataFrame({
        'edit_dist': [s['edit_distance_AB'] for s in test_samples_meta],
        'y_true':       y_true,
        'y_pred':       y_pred,
        'y_pred_prob':  y_pred_prob # 0.5に近いほどモデルが迷った
        })

    # 正誤判定と距離の差を追加
    analysis_df['is_correct'] = (analysis_df['y_true'] == analysis_df['y_pred'])

    # CSVとして保存
    analysis_df.to_csv(os.path.join(save_dir, "detailed_analysis.csv"), index=False)

    # --- サマリーレポートの作成 ---
    with open(os.path.join(save_dir, "fixed_seqA_summary.txt"), "w") as f:
        f.write(f"Experiment Name: {EXPERIMENT_NAME}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write("-" * 30 + "\n")
        f.write(f"ACC: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"P  : {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"R  : {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1 : {f1_score(y_true, y_pred):.4f}\n")
        f.write(f"AUC: {roc_auc_score(y_true, y_pred_prob):.4f}\n")
        # 距離の差が小さい（難易度が高い）ケースでの精度を自動計算して出力
        # hard_cases = analysis_df[analysis_df['dist_diff'] <= 2]
        # f.write(f"Accuracy (Hard cases: dist_diff <= 2): {hard_cases['is_correct'].mean():.4f}\n")


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    print("ACC:", accuracy_score(y_true, y_pred))
    print("P  :", precision_score(y_true, y_pred))
    print("R  :", recall_score(y_true, y_pred))
    print("F1 :", f1_score(y_true, y_pred))
    # 参考で AUC（距離差 dAC - dAB をスコアに）
    print("AUC:", roc_auc_score(y_true, y_pred_prob))
    # 統計的評価
    # --- 準備：難易度をバケット（0-1, 2-5...）に分ける ---
    bins = [0, threshold_T-20, threshold_T-10, threshold_T, threshold_T+10, threshold_T+20, np.inf]
    labels = ['0~median-20 (Easy)', 'median-20~median-10', 'median-10~median (Hard)', 'median+1~median+10', 'median+11~median+20', 'median+21~']
    analysis_df['diff_bucket'] = pd.cut(analysis_df['edit_dist'], bins=bins, labels=labels, include_lowest=True)

    # バケットごとに正解率（mean）を計算
    bucket_stats = analysis_df.groupby('diff_bucket', observed=True)['is_correct'].mean()    
    # --- 描画 ---
    plt.figure(figsize=(8, 5))
    plt.bar(bucket_stats.index, bucket_stats.values, color='skyblue', edgecolor='black')

    plt.title("Accuracy by Distance Difference (Difficulty)")
    plt.xlabel("Distance Difference (|dist_AB - dist_AC|)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.axhline(0.5, color='red', linestyle='--', label='Random Guess (0.5)') # 期待値の線
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # ディレクトリ内に保存
    plt.savefig(os.path.join(save_dir, "analysis_difficulty_bar.png"))
    plt.close()

    # --- 準備：難易度をバケット（0-1, 2-5...）に分ける ---
    bins = [0, threshold_T-20, threshold_T-10, threshold_T, threshold_T+10, threshold_T+20, np.inf]
    labels = ['0~median-20 (Easy)', 'median-20~median-10', 'median-10~median (Hard)', 'median+1~median+10', 'median+11~median+20', 'median+21~']
    analysis_df['diff_bucket'] = pd.cut(analysis_df['edit_dist'], bins=bins, labels=labels, include_lowest=True)

    # バケットごとに正解率（mean）を計算
    bucket_stats = analysis_df.groupby('diff_bucket', observed=True)['is_correct'].count()    
    # --- 描画 ---
    plt.figure(figsize=(8, 5))
    plt.bar(bucket_stats.index, bucket_stats.values, color='skyblue', edgecolor='black')

    plt.title("Accuracy by Distance Difference (Difficulty)")
    plt.xlabel("Distance Difference (|dist_AB - dist_AC|)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 10000)
    plt.axhline(0.5, color='red', linestyle='--', label='Random Guess (0.5)') # 期待値の線
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # ディレクトリ内に保存
    plt.savefig(os.path.join(save_dir, "analysis_difficulty_count_bar.png"))
    plt.close()

    # --- 描画 ---
    plt.figure(figsize=(8, 6))
    # 正解は青、不正解は赤でプロット
    correct_data = analysis_df[analysis_df['is_correct'] == True]
    wrong_data = analysis_df[analysis_df['is_correct'] == False]

    plt.scatter(correct_data['edit_dist'], correct_data['y_pred_prob'], 
                alpha=0.3, color='blue', label='Correct', s=10)
    plt.scatter(wrong_data['edit_dist'], wrong_data['y_pred_prob'], 
                alpha=0.3, color='red', label='Wrong', s=10)

    plt.title("Distance Diff vs Model Score")
    plt.xlabel("edit_distance_AB")
    plt.ylabel("Model Score (y_pred_prob)")
    plt.axhline(0, color='black', lw=1) # 判定の境界線
    plt.legend()

    plt.savefig(os.path.join(save_dir, "analysis_score_scatter.png"))
    plt.close()

    print("correct_data mean", (correct_data['y_pred_prob'] - threshold_T).abs().mean())
    print("wrong_data mean", (wrong_data['y_pred_prob'] - threshold_T).abs().mean())