import os
import pickle
import hashlib
import numpy as np
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

# Constants
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
CACHE_VERSION = 2  # 当数据结构变更时更新此版本号


def read_dataset(args, path):
    """带缓存的多进程数据加载"""
    # 生成缓存文件名
    cache_file = generate_cache_file(args, path)
    
    # 尝试加载缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"Loading cached dataset from {cache_file}")
                return pickle.load(f)
        except Exception as e:
            print(f"Cache load failed ({e}), regenerating...")

    # 处理数据集
    dataset = process_dataset(args, path)
    
    # 保存缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved processed dataset to {cache_file}")
    
    return dataset

def generate_cache_file(args, path):
    """生成唯一的缓存文件名"""
    # 基础参数
    # base_name = os.path.basename(path).split('.')[0]
    base_name = path.split('.')[0]
    param = f"{args.seq_length}_{args.soft_targets}"
    param_hash = hashlib.md5(param.encode()).hexdigest()[:8]
    
    # Tokenizer特征（示例使用词汇表大小）
    vocab_hash = str(len(args.tokenizer.vocab))[:6]
    
    return f"{base_name}_seq{args.seq_length}_{param_hash}_{vocab_hash}.pkl"

def process_dataset(args, path):
    num_workers = min(mp.cpu_count(), 8)  # 根据CPU核心数自动调整
    chunk_size = 1000  # 每个进程处理的行数

    # 获取文件总行数（用于进度条）
    total_lines = sum(1 for _ in open(path, 'r', encoding='utf-8')) - 1  # 减去header

    # 创建进程池
    with mp.Pool(processes=num_workers) as pool:
        # 任务生成器
        task_generator = generate_tasks(path, chunk_size)
        
        # 并行处理
        results = []
        with tqdm(total=total_lines, desc='Processing') as pbar:
            for batch in pool.imap_unordered(
                partial(process_chunk, args=args), 
                task_generator
            ):
                results.extend(batch)
                pbar.update(len(batch))
                
    return results

def generate_tasks(path, chunk_size):
    """生成数据块任务"""
    with open(path, 'r', encoding='utf-8') as f:
        # 跳过header
        header = next(f)
        columns = {name: i for i, name in enumerate(header.strip().split('\t'))}
        
        # 生成数据块
        chunk = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield (columns, chunk)
                chunk = []
        if chunk:
            yield (columns, chunk)

def process_chunk(task, args):
    """进程工作函数"""
    columns, lines = task
    seq_len = args.seq_length
    tokenizer = args.tokenizer  # 每个进程独立初始化
    
    # 预分配内存
    batch_size = len(lines)
    token_buff = np.zeros((batch_size, seq_len), dtype=np.int32)
    seg_buff = np.zeros((batch_size, seq_len), dtype=np.int8)
    labels = np.zeros(batch_size, dtype=np.int32)
    logits_list = []
    
    has_text_b = "text_b" in columns
    has_logits = args.soft_targets and "logits" in columns
    
    for i, line in enumerate(lines):
        parts = line.split('\t')
        
        # Tokenization
        text_a = parts[columns["text_a"]]
        text_b = parts[columns["text_b"]] if has_text_b else None
        
        # 使用自定义tokenizer处理
        tokens_a = [CLS_TOKEN] + tokenizer.tokenize(text_a)
        if has_text_b:
            tokens_b = tokenizer.tokenize(text_b) + [SEP_TOKEN]
            tokens = tokens_a + tokens_b
        else:
            tokens = tokens_a
        
        # 转为IDs并截断/填充
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > seq_len:
            token_ids = smart_truncate(token_ids, seq_len, has_text_b)
        elif len(token_ids) < seq_len:
            token_ids += [0] * (seq_len - len(token_ids))
        
        # 填充缓冲区
        token_buff[i] = token_ids
        
        # 生成segments
        seg_ids = generate_segment_ids(
            tokens_a, 
            tokens_b if has_text_b else None, 
            seq_len
        )
        seg_buff[i] = seg_ids
        
        # 标签处理
        labels[i] = int(parts[columns["label"]])
        if has_logits:
            logits = list(map(float, parts[columns["logits"]].split()))
            logits_list.append(logits)
    
    # 组装结果批次
    if has_logits:
        return list(zip(token_buff, labels, seg_buff, logits_list))
    else:
        return list(zip(token_buff, labels, seg_buff))

def smart_truncate(token_ids, seq_len, has_text_b):
    """智能截断策略"""
    if has_text_b:
        split_point = len(token_ids) // 2
        keep = seq_len // 2
        return token_ids[:keep] + token_ids[-keep:]
    else:
        return token_ids[:seq_len]

def generate_segment_ids(tokens_a, tokens_b, seq_len):
    """生成分段标识"""
    seg_ids = np.zeros(seq_len, dtype=np.int8)
    a_len = min(len(tokens_a), seq_len)
    seg_ids[:a_len] = 1
    
    if tokens_b is not None:
        b_len = min(len(tokens_b), seq_len - a_len)
        seg_ids[a_len:a_len+b_len] = 2
    
    return seg_ids

