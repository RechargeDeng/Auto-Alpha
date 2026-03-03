import os
import pandas as pd
import time
import pickle   
import re
from collections import defaultdict


def tradeday_slelect(path):
    """
    读取路径下所有形如merged_date.csv的文件，并提取所有的date，组成一个列表并返回
    """
    if not os.path.exists(path):
        return []
    
    dates = []
    for filename in os.listdir(path):
        if filename.endswith('.pq'):
            # 提取文件名中的日期部分
            date_part = filename.replace('.pq', '')
            dates.append(date_part)
        elif filename.endswith('.parquet'):
            date_part = filename.replace('.parquet', '')
            date_part = date_part.split('_')[-1]
            dates.append(date_part)
    
    return sorted(dates)

def find_outer_brace(s: str):
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return start, i
    return None


def split_by_top_level_commas(s: str):
    parts = []
    depth = 0
    last = 0
    for i, ch in enumerate(s):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(s[last:i].strip())
            last = i + 1
    parts.append(s[last:].strip())
    return parts


def is_pure_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_formula_collect_atoms(s: str, atoms: list):
    s = s.strip()
    brace = find_outer_brace(s)

    # 情况 1：已经没有花括号了
    if brace is None:
        if ',' in s:
            for part in s.split(','):
                part = part.strip()
                if part and not is_pure_number(part):
                    atoms.append(part)
        else:
            if s and not is_pure_number(s):
                atoms.append(s)
        return

    # 情况 2：还有花括号，继续按原逻辑递归
    start, end = brace
    inner = s[start + 1:end]
    parts = split_by_top_level_commas(inner)

    for part in parts:
        parse_formula_collect_atoms(part, atoms)


def generate_alpha_backtest_command(fml_list, start_date, end_date, update_path):
    commands = []
    for fml in fml_list:
        atoms = []
        parse_formula_collect_atoms(fml, atoms)
        atoms = list(set(atoms))
        fields_args = " ".join(f"'{f}'" for f in atoms)
        command = f"~/miniconda3/envs/ml/bin/python /titan_gluster/bdeng/High-Frequency-Predictor/alpha_ge.py --data_path /titan_gluster/bdeng/data/puneet_32_new --start_date {start_date} --end_date {end_date} --predict_interval 48 --fields {fields_args} --fml_list '{fml}' --warmup_interval 480 --vs_index True --use_cython True --is_update True --update_path {update_path}  --limit_mask_path /titan_gluster/bdeng/data/limit_mask/update_ver1 --if_barra True"
        commands.append(command)
    output_file = os.path.join('/titan_gluster/bdeng/command', f"factor_backtest.txt")
    # 写入文件
    with open(output_file, 'w') as f:
        for command in commands:
            f.write(command + '\n')
    # 读取sbatch.sh模板文件
    sbatch_template_path = '/titan_gluster/bdeng/alpha_pre/sbatch.sh'
    with open(sbatch_template_path, 'r') as f:
        sbatch_content = f.read()
    
    # 修改第二行的TITAN_SCRIPT路径
    lines = sbatch_content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('TITAN_SCRIPT='):
            lines[i] = f'TITAN_SCRIPT={output_file}'
            break
    for i, line in enumerate(lines):
        if line.startswith('  --mem='):
            lines[i] = f'  --mem=24G \\'
            break
    # 重新组合内容
    modified_sbatch_content = '\n'.join(lines)
    
    # 创建sbatch文件
    sbatch_output_file = os.path.join('/titan_gluster/bdeng/command', f"factor_backtest.sh")
    with open(sbatch_output_file, 'w') as f:
        f.write(modified_sbatch_content)
    return output_file

def generate_alpha_dump_command(fml_list, datelist):
    commands = []
    for fml in fml_list:
        for date in datelist:
            atoms = []
            parse_formula_collect_atoms(fml, atoms)
            atoms = list(set(atoms))
            fields_args = " ".join(f"'{f}'" for f in atoms)
            command = f"~/miniconda3/envs/ml/bin/python /titan_gluster/bdeng/High-Frequency-Predictor/alpha_ge.py --data_path /titan_gluster/bdeng/data/puneet_32_new --start_date {date} --end_date {date} --predict_interval 48 --fields {fields_args} --fml_list '{fml}' --warmup_interval 480 --vs_index True --use_cython True --is_dump True --dump_folder quantile"
            commands.append(command)
    output_file = os.path.join('/titan_gluster/bdeng/command', f"factor_dump.txt")
    # 写入文件
    with open(output_file, 'w') as f:
        for command in commands:
            f.write(command + '\n')
    # 读取sbatch.sh模板文件
    sbatch_template_path = '/titan_gluster/bdeng/alpha_pre/sbatch.sh'
    with open(sbatch_template_path, 'r') as f:
        sbatch_content = f.read()
    
    # 修改第二行的TITAN_SCRIPT路径
    lines = sbatch_content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('TITAN_SCRIPT='):
            lines[i] = f'TITAN_SCRIPT={output_file}'
            break
    for i, line in enumerate(lines):
        if line.startswith('  --mem='):
            lines[i] = f'  --mem=2G \\'
            break
    # 重新组合内容
    modified_sbatch_content = '\n'.join(lines)
    
    # 创建sbatch文件
    sbatch_output_file = os.path.join('/titan_gluster/bdeng/command', f"factor_dump.sh")
    with open(sbatch_output_file, 'w') as f:
        f.write(modified_sbatch_content)
    return output_file

def generate_merge_command(date_list,dump_folder):
    commands = []
    for date in date_list:
        command = f"python3.11 /titan_gluster/bdeng/crontab/merge.py --base_path /titan_gluster/bdeng/data/factors_new/daily_store --dump_folder {dump_folder} --date {date}"
        commands.append(command)

    # 创建输出文件路径
    output_file = os.path.join('/titan_gluster/bdeng/command', f"factor_merge_quantile.txt")
    
    # 写入文件
    with open(output_file, 'w') as f:
        for command in commands:
            f.write(command + '\n')
    # 读取sbatch.sh模板文件
    sbatch_template_path = '/titan_gluster/bdeng/alpha_pre/sbatch.sh'
    with open(sbatch_template_path, 'r') as f:
        sbatch_content = f.read()
    
    # 修改第二行的TITAN_SCRIPT路径
    lines = sbatch_content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('TITAN_SCRIPT='):
            lines[i] = f'TITAN_SCRIPT={output_file}'
            break
    for i, line in enumerate(lines):
        if line.startswith('  --mem='):
            lines[i] = f'  --mem=4G \\'          ####################################
            break
    # 重新组合内容
    modified_sbatch_content = '\n'.join(lines)
    
    # 创建sbatch文件
    sbatch_output_file = os.path.join('/titan_gluster/bdeng/command', f"factor_merge_quantile.sh")
    with open(sbatch_output_file, 'w') as f:
        f.write(modified_sbatch_content)
    return output_file

def run_backtest(exprs):
    start_date = '20240104'
    end_date = '20241231'
    generate_alpha_backtest_command(exprs, start_date, end_date, '/titan_gluster/bdeng/data/stats/2025/auto_alpha_backtest_ver01.csv')
    os.system('sh /titan_gluster/bdeng/command/factor_backtest.sh')
    time.sleep(5)  # 等待一会儿，确保文件写入
    result_df = pd.read_csv('/titan_gluster/bdeng/data/stats/2025/auto_alpha_backtest_ver01.csv')
    return result_df

# def _main():
#     start_date = '20260105'
#     end_date = '20260123'
#     datelist = tradeday_slelect('/titan_gluster/bdeng/data/puneet_32_new')
#     datelist = datelist[datelist.index(start_date):datelist.index(end_date)+1]
#     # stats_df = pd.read_csv('/titan_gluster/bdeng/data/stats/2025/ai_formula.csv')
#     # stats_df = stats_df[(stats_df.filter(like="long_alpha_sharpe") > 3.0).any(axis=1)]
#     # alphas = list(set(stats_df['fml'].to_list()))
#     #alphas = alpha_stability + alpha_imbalance + alpha_line + alpha_large + alpha_other_0 + alpha_other_1 + alpha_conditional

#     # cols = pd.read_parquet('/titan_gluster/bdeng/data/puneet_32_new/20230106.pq').columns.to_list()
#     # cols = [xx for xx in cols if ('ChangeRate' not in xx) and ('datetime' not in xx) and('ticker' not in xx) and ('SpreadOrder.OrderPriceMean' not in xx) and ('TradeLine.ClosePrice' not in xx)]
#     # cols = [xx for xx in cols if "Stability" in xx or 'Skew' in xx]
#     # alphas = []
#     # for feilds in cols:
#     #     alphas += formula_ge_0(feilds)

#     # stats_df = pd.read_csv('/titan_gluster/bdeng/data/stats/2025/specify_formula_0.csv')
#     # stats_df = stats_df[(stats_df.filter(like="long_alpha_sharpe") > 0.8).any(axis=1)]
#     # alphas = list(set(stats_df['fml'].to_list()))
#     # generate_alpha_backtest_command(alphas, start_date, end_date)
#     # os.system('sh /titan_gluster/bdeng/command/factor_backtest.sh')

#     stats_df = pd.read_csv('/titan_gluster/bdeng/data/stats/2025/specify_formula_1.csv')
#     stats_df_2half = pd.read_csv('/titan_gluster/bdeng/data/stats/2025/specify_formula_0.csv')
#     stats_df = stats_df[(stats_df.filter(like="long_alpha_sharpe") > 3.5).any(axis=1)]
#     stats_df_2half = stats_df_2half[(stats_df_2half.filter(like="long_alpha_sharpe") > 1.7).any(axis=1)]
#     stats_df_2half = stats_df_2half[stats_df_2half['fml'].isin(stats_df['fml'])]
#     alphas = list(set(stats_df_2half.sort_values(['long_alpha_sharpe'], ascending=False).head(50)['fml'].tolist()))
#     generate_alpha_dump_command(alphas, datelist)
#     os.system('sh /titan_gluster/bdeng/command/factor_dump.sh')
#     generate_merge_command(datelist,'quantile')
#     os.system('sh /titan_gluster/bdeng/command/factor_merge_quantile.sh')
# if __name__ == "__main__":
#     _main()

# ~/miniconda3/envs/ml/bin/python /titan_gluster/bdeng/High-Frequency-Predictor/alpha_ge.py --data_path /titan_gluster/bdeng/data/puneet_32_new --start_date 20250102 --end_date 20251231 --predict_interval 48 --fields 'SpreadCancel.Volume_Stability' --fml_list 'tsquantile{SpreadCancel.Volume_Stability,120,0.2}' --warmup_interval 480 --vs_index True --use_cython True --is_dump True --dump_folder test
# nohup python /titan_gluster/bdeng/High-Frequency-Predictor/ai_alpha/ai_alpha_ge.py > /titan_gluster/bdeng/High-Frequency-Predictor/ai_alpha/ai_alpha_ge.out 2>&1 &
# ps -ef | grep ai_alpha_ge.py
# tail -f ai_alpha_ge.out
# pkill -f ai_alpha_ge.py
# kill -9 PID