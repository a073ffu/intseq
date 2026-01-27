import generate_program_ver2 as generate_program
import program_ver2 as program
import weight_ver2 as weight

import numpy as np
import random
import string
import sys
from typing import Optional, List, Tuple, Dict, Union
import time

CHECK_CONSTANT_N = 10 

# 数列データ生成関数
def generate_initial_sequence_sample(depth=10, numeric_sequence_length=20)->dict:
    """
    1つ目の数列データと関連情報を生成します。
    オーバーフロー、NaN/Inf、定数列、等差数列の場合は再試行します。
    生成されたプログラムの実際の深さを 'program_depth' として返します。
    """
    max_attempts = 1000 
    attempt_count = 0

    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            # generate_initial_sequence_sampleに渡されたdepthをProgramGeneratorのmax_depthとして使用
            # depthがデフォルトの場合 (10)、random.randint(1,10)の深さのツリーが生成される
            actual_program_depth = random.randint(1, depth) # この深さが実際に生成される深さ
            program_gen_1 = generate_program.ProgramGenerator(actual_program_depth)
            program_gen_1.build_tree() 
            
            necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(program_gen_1.root_node, numeric_sequence_length)
            x = list(range(1, necessary_numeric_sequence_length_1 + 1)) 

            program_inter_1 = program.ProgramInterpreter(program_gen_1.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_1)
            program_executor_1 = program_inter_1.build()
            
            seq_1 = program_executor_1[0].calc(x) 
            
            # --- 数列の品質チェック ---
            if not seq_1 or any(np.isinf(val) or np.isnan(val) for val in seq_1):
                raise program.SequenceError("Generated sequence 1 is empty or contains Inf/NaN.")            
            if program.check_if_constant_sequence(seq_1[0:numeric_sequence_length], n=CHECK_CONSTANT_N): 
                raise program.SequenceError(f"Generated sequence 1 has {CHECK_CONSTANT_N} consecutive constant elements at head/tail.")
            if program.check_if_arithmetic_progression(seq_1[0:20]):
                raise program.SequenceError("Generated sequence 1 is an arithmetic progression.")

            information_amount_1 = program_gen_1.get_information_amount()
            
            return {
                'numeric_sequence_1': seq_1[0:20],
                'token_sequence_1': program_gen_1.get_token_sequence(),
                'initial_sequence_info_amount': information_amount_1, 
                'initial_sequence_depth': actual_program_depth 
            }
        except program.SequenceError as e:
            continue 
        except Exception as e: 
            # 【変更】エラー内容を表示するようにする
            print(f"DEBUG ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"ERROR: Failed to generate a valid initial sequence after {max_attempts} attempts. Check program generation logic.")
    return None 

# 与えられた数列データに2つ目の数列データを追加する関数
# 変更点: 戻り値に 'program_depth' を追加
def generate_dependent_sequence_sample(data:dict, depth:int, numeric_sequence_length=20, is_x_bounded = None)->dict:
    """
    与えられた1つ目の数列データに、2つ目の数列データと依存関係情報を追加します。
    オーバーフロー、NaN/Inf、定数列、等差数列の場合は再試行します。
    2つ目のプログラムの実際の深さを 'program_depth' として返します。
    """
    max_attempts = 1000 
    attempt_count = 0

    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            program_gen_2 = generate_program.ProgramGenerator(depth) # depth引数が ProgramGenerator の max_depth として使われる
            program_gen_2.build_tree() 

            generated_is_x_bounded = program_gen_2.check_is_x_bounded()

            if is_x_bounded is not None and is_x_bounded != generated_is_x_bounded:
                continue 

            necessary_numeric_sequence_length_2 = generate_program.calculate_original_sequence_length(program_gen_2.root_node, numeric_sequence_length)

            program_gen_1_temp = generate_program.ProgramGenerator(depth) 
            program_gen_1_temp.convert_token_sequence_to_tree(data['token_sequence_1'])
            necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(program_gen_1_temp.root_node, necessary_numeric_sequence_length_2)

            x_initial = list(range(1, necessary_numeric_sequence_length_1 + 1)) 
            
            program_inter_1 = program.ProgramInterpreter(data['token_sequence_1'], numeric_sequence_length=necessary_numeric_sequence_length_1)
            program_executor_1 = program_inter_1.build()
            seq_1 = program_executor_1[0].calc(x_initial) 

            program_inter_2 = program.ProgramInterpreter(program_gen_2.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_2)
            program_executor_2 = program_inter_2.build()
            seq_2 = program_executor_2[0].calc(seq_1) 
            
            # --- 数列の品質チェック (seq_2の数列を生成するプログラムをチェック) ---
            if not seq_2 or any(np.isinf(val) or np.isnan(val) for val in seq_2):
                 raise program.SequenceError("Generated sequence 2 is empty or contains Inf/NaN.")
            if program.check_if_constant_sequence(program_executor_2[0].calc(list(range(1, necessary_numeric_sequence_length_2 + 1)))[0:numeric_sequence_length], n=CHECK_CONSTANT_N):
                raise program.SequenceError(f"Generated sequence 2 has {CHECK_CONSTANT_N} consecutive constant elements at head/tail.")
            if program.check_if_arithmetic_progression(program_executor_2[0].calc(list(range(1, necessary_numeric_sequence_length_2 + 1)))[0:numeric_sequence_length]):
                raise program.SequenceError("Generated sequence 2 is an arithmetic progression.")

            information_amount_2 = program_gen_2.get_information_amount()

            data_updated = data.copy() 
            data_updated.update(
                {
                    'numeric_sequence_2': seq_2[0:20], 
                    'token_sequence_2': program_gen_2.get_token_sequence(),
                    'transformed_sequence_info_amount': information_amount_2, # ★ ここを修正 ★
                    'is_x_bounded': int(generated_is_x_bounded),
                    'transformed_sequence_depth': program_gen_2.max_depth # ★ program_depth_2 も一貫性を持たせるため変更推奨 ★
                }
            )
            '''コード変更により不要
            # 依存数列の順序変更 (ランダムにseq1とseq2をスワップ)
            # 複雑度情報 (initial_sequence_info_amount, transformed_sequence_info_amount) はスワップしないようにする
            if random.random() < 0.5:
                temp_numeric_sequence_1 = data_updated['numeric_sequence_1']
                temp_token_sequence_1 = data_updated['token_sequence_1']
                # ★ ここは以前削除されたはずの情報量スワップの行。再追加しない。★
                # temp_initial_sequence_info_amount = data_updated['initial_sequence_info_amount']
                temp_initial_sequence_depth = data_updated['initial_sequence_depth'] # ★ テンポラリ変数名も変更 ★

                data_updated['numeric_sequence_1'] = data_updated['numeric_sequence_2']
                data_updated['token_sequence_1'] = data_updated['token_sequence_2']
                # ★ ここは以前削除されたはずの情報量スワップの行。再追加しない。★
                # data_updated['initial_sequence_info_amount'] = data_updated['transformed_sequence_info_amount']
                data_updated['initial_sequence_depth'] = data_updated['transformed_sequence_depth'] # ★ キー名変更 ★

                data_updated['numeric_sequence_2'] = temp_numeric_sequence_1
                data_updated['token_sequence_2'] = temp_token_sequence_1
                # ★ ここは以前削除されたはずの情報量スワップの行。再追加しない。★
                # data_updated['transformed_sequence_info_amount'] = temp_initial_sequence_info_amount
                data_updated['transformed_sequence_depth'] = temp_initial_sequence_depth # ★ キー名変更 ★

                data_updated.update({'was_swapped':1})
            else:
                data_updated.update({'was_swapped':0})
            '''
            return data_updated

        except program.SequenceError as e:
            continue 
        except Exception as e: 
            # 【変更】エラー内容を表示するようにする
            print(f"DEBUG ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"ERROR: Failed to generate a valid dependent sequence after {max_attempts} attempts for initial data: {data['token_sequence_1']}. Check program generation logic.")
    return None 

# 既存の generate_classification_data は変更なし
def generate_classification_data(depth:int = None, num_samples=10000):
    """
    指定された深さまたはランダムな深さで、依存関係のある/ない数列ペアデータセットを生成します。
    無効な（オーバーフロー、NaN/Inf、定数列、等差数列の）サンプルはスキップされ、目標のサンプル数を生成します。
    """
    print(f"[DEBUG] Starting generate_classification_data for {num_samples} samples.")
    
    generated_data = []
    current_dependent_count = 0
    current_independent_count = 0
    
    sample_generation_attempts = 0 
    max_total_attempts = num_samples * 5 
    
    while len(generated_data) < num_samples and sample_generation_attempts < max_total_attempts:
        sample_generation_attempts += 1
        
        # generate_initial_sequence_sample には depth 引数が渡されないので、デフォルトの10が使われ、
        # 内部で random.randint(1,10) の深さのツリーが生成される。
        initial_sample = generate_initial_sequence_sample() 
        
        if initial_sample is None: 
            continue

        if current_dependent_count < num_samples / 2 and current_independent_count < num_samples / 2:
            is_bounded_request = random.choice([True, False])
        elif current_dependent_count < num_samples / 2:
            is_bounded_request = True 
        else:
            is_bounded_request = False 

        # generate_dependent_sequence_sample の depth 引数には、generate_classification_data の depth 引数 (またはデフォルト) を渡す。
        # generate_classification_data(depth=D) で呼び出された場合、ProgramGenerator(D) が使われる。
        # generate_classification_data() と呼び出された場合、random.randint(3,8) が使われる。
        # ここは、generate_classification_data が ProgramGenerator の max_depth を制御するポイント。
        dependent_sample = generate_dependent_sequence_sample(
            data=initial_sample, 
            depth=depth if depth is not None else random.randint(3,8), # ★ ここは generate_classification_data の引数に従う ★
            is_x_bounded=is_bounded_request
        )

        if dependent_sample is not None:
            generated_data.append(dependent_sample)
            if dependent_sample['is_x_bounded'] == 1:
                current_dependent_count += 1
            else:
                current_independent_count += 1
            print(f"[DEBUG] Generated {len(generated_data)}/{num_samples} samples (Dep: {current_dependent_count}, Indep: {current_independent_count}) (Total attempts: {sample_generation_attempts})")
        else:
            pass 

    if len(generated_data) < num_samples:
        print(f"WARNING: Could not generate {num_samples} samples after {max_total_attempts} total attempts. Generated only {len(generated_data)} samples.")
    
    print(f"[DEBUG] Finished generate_classification_data. Total dependent sequences: {current_dependent_count}. Total samples: {len(generated_data)}")
    return generated_data


# ★ 新しく追加する _generate_single_sample_by_info_range ヘルパー関数 ★
def _generate_single_sample_by_info_range(
    target_info_min: float, target_info_max: float, 
    is_x_bounded_request: bool, 
    max_build_attempts_per_sample: int = 1000, # 1つの有効なサンプルを生成するための最大試行回数
    max_program_depth: int = 15 # プログラム生成時の最大許容深度
) -> Optional[dict]:
    """
    指定された情報量の範囲と依存関係の有無を満たす単一の数列ペアサンプルを生成します。
    目標情報量に到達するまでランダムな深さのプログラム生成を試行します。
    """
    attempt_count = 0
    
    while attempt_count < max_build_attempts_per_sample:
        attempt_count += 1
        
        try:
            # 1つ目の数列をランダムな深さで生成 (generate_initial_sequence_sample のデフォルト深度を使用)
            # generate_initial_sequence_sample には ProgramGenerator(random.randint(1,10)) の深さが使われる
            initial_sample = generate_initial_sequence_sample() 
            if initial_sample is None: 
                continue # initial_sample生成失敗

            # 2つ目の数列のプログラムは、指定された情報量の範囲に収まるように生成
            # ProgramGenerator の max_depth をランダムに設定して build_tree を呼び出す (強制はしない)
            # 複雑度指定のツリー生成は、深さの範囲内でランダムに生成される
            actual_program_depth_for_p2 = random.randint(1, max_program_depth) # プログラム2の深さの上限をランダムに設定

            temp_program_gen_2 = generate_program.ProgramGenerator(actual_program_depth_for_p2)
            temp_program_gen_2.build_tree() # ランダム深さのツリーを生成

            generated_info_amount_2 = temp_program_gen_2.get_information_amount()
            generated_is_x_bounded_2 = temp_program_gen_2.check_is_x_bounded()

            # 情報量の範囲と依存関係の有無の条件をチェック
            if not (target_info_min <= generated_info_amount_2 <= target_info_max):
                continue # 情報量が範囲外なら再試行
            if is_x_bounded_request is not None and is_x_bounded_request != generated_is_x_bounded_2:
                continue # 依存関係の有無が一致しないなら再試行

            # フィルタリングを通過したら、generate_dependent_sequence_sample を呼び出す
            # generate_dependent_sequence_sample は ProgramGenerator に max_depth を渡すため、
            # ここでは actual_program_depth_for_p2 を渡す
            final_sample = generate_dependent_sequence_sample(
                initial_sample, 
                depth=actual_program_depth_for_p2, 
                is_x_bounded=is_x_bounded_request # _generate_single_sample_by_info_range の is_x_bounded_request を渡す
            )
            
            if final_sample is not None:
                # 生成されたサンプルに、2つ目のプログラムの深さ情報を追加
                # generate_dependent_sequence_sample ですでに 'program_depth_2' が追加されているはず
                return final_sample
            
        except program.SequenceError:
            pass # スキップして再試行
        except Exception as e:
            # print(f"DEBUG: Error during single sample generation attempt {attempt_count}: {type(e).__name__}: {e}")
            pass # スキップして再試行

    print(f"ERROR: Failed to generate a valid sample in info range [{target_info_min}, {target_info_max}] after {max_build_attempts_per_sample} attempts.")
    return None


def generate_by_information_amount(
    info_ranges: List[Tuple[float, float]], 
    num_samples_per_range: int, 
    dataset_type: str = "classification", 
    max_program_depth: int = 15 # プログラム生成時の最大許容深度
) -> Dict[Tuple[float, float], List[dict]]:
    """
    指定された複数の情報量範囲ごとにデータセットを生成し、振り分けます。
    生成された全ての有効なプログラムを、該当する情報量範囲のリストに振り分けます。
    
    Args:
        info_ranges (List[Tuple[float, float]]): 生成したい情報量範囲のリスト [(min1, max1), (min2, max2), ...]
        num_samples_per_range (int): 各情報量範囲で生成する目標サンプル数
        dataset_type (str): "classification" または "regression" (依存関係の有無のバランス)
        max_program_depth (int): プログラム生成時の最大許容深度。
    
    Returns:
        Dict[Tuple[float, float], List[dict]]: 各情報量範囲に対応する生成済みデータセットの辞書。
    """
    print(f"\n--- Starting Data Generation and Distribution by Information Amount Range ---")
    print(f"Generating {num_samples_per_range} samples per range for {len(info_ranges)} ranges.")

    # 各情報量範囲のデータリストを初期化
    all_generated_datasets = {info_range: [] for info_range in info_ranges}
    
    # 各情報量範囲の生成済みサンプル数を追跡
    generated_counts_per_range = {info_range: 0 for info_range in info_ranges}

    total_generated_samples = 0
    total_required_samples = len(info_ranges) * num_samples_per_range
    total_attempts = 0
    
    # 最大試行回数 (データ生成全体の制限)
    # 複雑度範囲が非常に細かく、有効なサンプルを見つけるのが難しい場合、多くする必要がある
    max_overall_attempts = total_required_samples * 10 # 目標サンプル総数の10倍を上限とする

    start_overall_time = time.time()

    # すべての範囲で目標サンプル数を満たすまでループ
    while total_generated_samples < total_required_samples and total_attempts < max_overall_attempts:
        total_attempts += 1
        
        # 1. プログラムの深さをランダムに選択
        # 情報量と深さは相関するため、広い深さの範囲を許容して様々な複雑度のプログラムを生成
        current_program_depth = random.randint(1, max_program_depth)

        # 2. 依存/非依存の要求を決定 (dataset_type に応じて)
        is_bounded_request = None 
        if dataset_type == "classification":
            # 分類データの場合、依存/非依存のバランスを取る (各範囲内でバランスを取るのは難しいので、全体で大まかにバランス)
            # ここでは単一サンプル生成なので、単純にランダムに要求
            is_bounded_request = random.choice([True, False]) 
            # より高度なバランス調整: 全体で依存/非依存の目標数を追跡し、不足している方を優先的に要求する

        elif dataset_type == "regression":
            is_bounded_request = True
        else:
            raise ValueError("Invalid dataset_type. Must be 'classification' or 'regression'.")
        
        # 3. 単一のサンプルを生成
        # _generate_single_sample_by_info_range は情報量範囲チェックなしで、有効なサンプルを生成するヘルパーとして利用
        # is_x_bounded_request は _generate_single_sample_by_info_range が is_x_bounded に従うように強制
        sample = _generate_single_sample_by_info_range(
            # ここでは情報量範囲は渡さない (フィルタリングは外で行う)
            target_info_min=0, # ダミー値 (フィルタリングは外で行うため)
            target_info_max=float('inf'), # ダミー値
            is_x_bounded_request=is_bounded_request,
            max_program_depth=current_program_depth # この深さで ProgramGenerator を初期化
        )

        if sample is not None:
            # 4. 生成されたサンプルの情報量を取得
            actual_info_amount = sample['transformed_sequence_info_amount'] 
            
            # 5. 生成されたサンプルを該当する情報量範囲のリストに振り分ける
            assigned_to_range = False
            for info_range in info_ranges:
                info_min, info_max = info_range
                
                # サンプルが現在の情報量範囲内にあり、かつその範囲の目標サンプル数に達していない場合
                if info_min <= actual_info_amount <= info_max and generated_counts_per_range[info_range] < num_samples_per_range:
                    all_generated_datasets[info_range].append(sample)
                    generated_counts_per_range[info_range] += 1
                    total_generated_samples += 1 # 全体で生成されたサンプル数をカウント
                    assigned_to_range = True
                    break # 1つの範囲に割り当てたら次のサンプルへ

            # 進捗表示 (適度な頻度で)
            if total_attempts % 1000 == 0 or total_generated_samples % (total_required_samples // 10) == 0:
                print(f"  Attempt {total_attempts}. Total Generated: {total_generated_samples}/{total_required_samples}. Remaining: {total_required_samples - total_generated_samples}.")
                # 各範囲の進捗も表示 (オプション)
                # for r, c in generated_counts_per_range.items(): print(f"    Range {r}: {c}/{num_samples_per_range}")

        # すべての範囲の目標サンプル数を満たしたか最終チェック
        if total_generated_samples >= total_required_samples:
            break

    end_overall_time = time.time()
    print(f"\n--- Data Generation Summary ---")
    print(f"Total time taken: {end_overall_time - start_overall_time:.2f} seconds ({(end_overall_time - start_overall_time) / 60:.2f} minutes).")
    print(f"Total attempts made: {total_attempts}.")
    print(f"Total samples generated across all ranges: {total_generated_samples}/{total_required_samples}.")

    # 目標サンプル数に達しなかった範囲を警告
    for info_range, count in generated_counts_per_range.items():
        if count < num_samples_per_range:
            print(f"WARNING: Range [{info_range[0]:.2f}, {info_range[1]:.2f}] only generated {count}/{num_samples_per_range} samples.")

    print(f"\n--- Finished Generating and Distributing Data by Information Amount Range ---")
    return all_generated_datasets