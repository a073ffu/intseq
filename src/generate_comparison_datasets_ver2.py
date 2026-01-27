import generate_program
import program_ver2 as program
import generate_data_set_ver2 as generate_data_set
import weight_ver2 as weight
import numpy as np
import random
import pickle
import os
import time

# --- 設定 ---
# ユーザーが数列Aを指定する場合、以下のリストに直接入力します。
# 例: SPECIFIC_SEQUENCE_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# 指定しない場合は None に設定します。
SPECIFIC_SEQUENCE_A = None

# 数列Aがランダム生成される場合の最大深度
RANDOM_A_MAX_DEPTH = 10 

# 数列B, C を生成するプログラムの深さの最大値 (変更点)
# generate_data_set.generate_dependent_sequence_sample の depth 引数に渡されるランダムな深さの上限
PROGRAM_BC_MAX_DEPTH = 10

# 各数列の最終的な表示長（データセットに保存される長さ）
NUMERIC_SEQUENCE_LENGTH = 20 

# 生成するサンプル数
NUM_SAMPLES_TO_GENERATE = 50000 

# 保存先のディレクトリ名
OUTPUT_DIR = "comparison_datasets"

# ファイル名
OUTPUT_FILENAME = f"comparison_data_ver3_randam_A.pkl"


# --- メイン処理 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    all_generated_comparison_samples = []
    generated_count = 0
    total_attempts = 0
    max_total_attempts = NUM_SAMPLES_TO_GENERATE * 10 # 無限ループ回避のための最大試行回数

    print("\n--- Starting Generation of Comparison Datasets ---")
    print(f"Generating samples where B and C programs have random depth between 1 and {PROGRAM_BC_MAX_DEPTH}")

    while generated_count < NUM_SAMPLES_TO_GENERATE and total_attempts < max_total_attempts:
        total_attempts += 1
        
        current_sample_data = {}

        try:
            # 1. 数列Aの生成 (ユーザー指定 or ランダム生成)
            if SPECIFIC_SEQUENCE_A is not None:
                # ユーザー指定の場合、プログラム生成情報等は含まない
                current_sample_data['numeric_sequence_A'] = SPECIFIC_SEQUENCE_A[0:NUMERIC_SEQUENCE_LENGTH]
                current_sample_data['program_A_token_sequence'] = ["x"]
                current_sample_data['program_A_initial_info_amount'] = None
                current_sample_data['program_A_initial_depth'] = None
                initial_sample_for_bc_gen = { # B,C生成のために必要な情報をダミーで作成
                    'numeric_sequence_1': SPECIFIC_SEQUENCE_A, 
                    'token_sequence_1': ["x"],
                    'information_amount_1': None, 
                    'initial_sequence_depth': None 
                }
            else:
                # ランダム生成の場合
                initial_sample_for_bc_gen = generate_data_set.generate_initial_sequence_sample(
                    depth=RANDOM_A_MAX_DEPTH, 
                    numeric_sequence_length=NUMERIC_SEQUENCE_LENGTH
                )
                if initial_sample_for_bc_gen is None:
                    continue # 数列Aの生成に失敗
                
                current_sample_data['numeric_sequence_A'] = initial_sample_for_bc_gen['numeric_sequence_1']
                current_sample_data['program_A_token_sequence'] = initial_sample_for_bc_gen['token_sequence_1']
                current_sample_data['program_A_initial_info_amount'] = initial_sample_for_bc_gen['initial_sequence_info_amount'] 
                current_sample_data['program_A_initial_depth'] = initial_sample_for_bc_gen['initial_sequence_depth'] 

            # ★ 数列B, C を生成するプログラムの深さをランダムに決定 (変更点) ★
            random_depth_for_BC_program = random.randint(1, PROGRAM_BC_MAX_DEPTH)

            # 2. 数列Bの生成 (数列Aを基に、依存関係あり)
            dependent_sample_B = generate_data_set.generate_dependent_sequence_sample(
                initial_sample_for_bc_gen, 
                depth=random_depth_for_BC_program, # ★ ランダムな深さを渡す ★
                numeric_sequence_length=NUMERIC_SEQUENCE_LENGTH,
                is_x_bounded=True # 依存関係ありを強制
            )
            if dependent_sample_B is None:
                continue # 数列Bの生成に失敗

            # 3. 数列Cの生成 (数列Aを基に、依存関係あり)
            # 数列Bとは異なるプログラムで生成されることを期待するため、再度ランダムな深さを決定
            random_depth_for_BC_program_C = random.randint(1, PROGRAM_BC_MAX_DEPTH) # C用にもランダムな深さ

            dependent_sample_C = generate_data_set.generate_dependent_sequence_sample(
                initial_sample_for_bc_gen, # 同じ数列Aを基にする
                depth=random_depth_for_BC_program_C, # ★ ランダムな深さを渡す ★
                numeric_sequence_length=NUMERIC_SEQUENCE_LENGTH,
                is_x_bounded=True # 依存関係ありを強制
            )
            if dependent_sample_C is None:
                continue # 数列Cの生成に失敗
            
            # 数列BとCが同じである場合はスキップ (同一ペアを避ける)
            if np.array_equal(dependent_sample_B['numeric_sequence_2'], dependent_sample_C['numeric_sequence_2']):
                continue

            # サンプルデータを構築
            current_sample_data.update({
                'numeric_sequence_B': dependent_sample_B['numeric_sequence_2'],
                'program_B_token_sequence': dependent_sample_B['token_sequence_2'],
                'program_B_transformed_info_amount': dependent_sample_B['transformed_sequence_info_amount'],
                'program_B_transformed_depth': dependent_sample_B['transformed_sequence_depth'], 

                'numeric_sequence_C': dependent_sample_C['numeric_sequence_2'],
                'program_C_token_sequence': dependent_sample_C['token_sequence_2'],
                'program_C_transformed_info_amount': dependent_sample_C['transformed_sequence_info_amount'],
                'program_C_transformed_depth': dependent_sample_C['transformed_sequence_depth'], 

                # 編集距離として information_amount を使用し、代入
                'edit_distance_AB': dependent_sample_B['transformed_sequence_info_amount'], 
                'edit_distance_AC': dependent_sample_C['transformed_sequence_info_amount'],
            })

            # target_label の決定
            # edit_distance_AB < edit_distance_AC なら 0, edit_distance_AB > edit_distance_AC なら 1
            # 等しい場合は、どちらかに割り振るか、スキップするか、ランダムにするか
            # ここでは、簡潔のため edit_distance_AB <= edit_distance_AC なら 0 とする（A,BがA,C以下なら0）
            if current_sample_data['edit_distance_AB'] <= current_sample_data['edit_distance_AC']:
                current_sample_data['target_label'] = 0
            else:
                current_sample_data['target_label'] = 1
            
            all_generated_comparison_samples.append(current_sample_data)
            generated_count += 1

            if generated_count % (NUM_SAMPLES_TO_GENERATE // 10) == 0:
                print(f"Generated {generated_count}/{NUM_SAMPLES_TO_GENERATE} comparison samples (Total attempts: {total_attempts})...")

        except program.SequenceError as e:
            # print(f"Skipping sample due to SequenceError: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error during sample generation (attempt {total_attempts}): {type(e).__name__}: {e}")
            # traceback.print_exc() # デバッグ用
            continue

    # 保存処理
    if generated_count < NUM_SAMPLES_TO_GENERATE:
        print(f"WARNING: Only {generated_count} comparison samples generated after {max_total_attempts} attempts.")
    else:
        print(f"Successfully generated {generated_count} comparison samples.")
    
    file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(all_generated_comparison_samples, f)
        print(f"Saved {generated_count} samples to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

    print("\n--- Generation finished ---")