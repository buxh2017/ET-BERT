import os
import mmap
import pickle
import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

# Constants
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
CACHE_VERSION = 1  # 缓存版本控制


def read_dataset(args, path):
    # # 没有多进程，spend time 30m
    dataset = []
    with open(path, mode="r", encoding="utf-8") as f:
        # 预加载列索引
        header = next(f).strip().split("\t")
        columns = {name: i for i, name in enumerate(header)}
        
        # 向量化预分配
        seq_len = args.seq_length
        cls_id = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])[0]
        sep_id = args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])[0] if SEP_TOKEN else 0

        # 批量处理参数
        batch_size = 1024  # 根据内存调整
        buffer = []
        
        for line in tqdm.tqdm(f, desc="Processing lines"):
            # 批量缓冲
            buffer.append(line)
            if len(buffer) < batch_size:
                continue
                
            # 处理批量
            batch = process_batch(buffer, args, columns, seq_len, cls_id, sep_id)
            dataset.extend(batch)
            buffer = []
        
        # 处理剩余数据
        if buffer:
            batch = process_batch(buffer, args, columns, seq_len, cls_id, sep_id)
            dataset.extend(batch)

    return dataset

def process_batch(lines, args, columns, seq_len, cls_id, sep_id):
    batch = []
    tokenizer = args.tokenizer
    has_text_b = "text_b" in columns
    label_idx = columns["label"]
    
    # 统一维度为 seq_len（而不是 max_tokens）
    token_buffer = np.zeros((len(lines), seq_len), dtype=np.int32)  # 修改点1
    seg_buffer = np.zeros((len(lines), seq_len), dtype=np.int8)     # 修改点2
    
    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        text_a = parts[columns["text_a"]]
        text_b = parts[columns["text_b"]] if has_text_b else None
        
        # Tokenization
        tokens_a = [CLS_TOKEN] + tokenizer.tokenize(text_a)
        if has_text_b:
            tokens_b = tokenizer.tokenize(text_b) + [SEP_TOKEN]
            tokens = tokens_a + tokens_b
        else:
            tokens = tokens_a
        
        # 转换为ID并处理长度
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > seq_len:
            # 智能截断策略
            if has_text_b:
                keep_a = min(len(tokens_a), int(seq_len * 0.4))
                keep_b = seq_len - keep_a
                token_ids = token_ids[:keep_a] + token_ids[-keep_b:]
            else:
                token_ids = token_ids[:seq_len]
        elif len(token_ids) < seq_len:
            # 统一填充
            token_ids += [0] * (seq_len - len(token_ids))
        
        # 写入token buffer
        token_buffer[i] = token_ids
        
        # 生成seg_ids（核心修正）
        seg_a_len = len(tokens_a) if has_text_b else len(tokens)
        seg_b_len = len(tokens_b) if has_text_b else 0
        
        # 动态调整实际有效长度
        valid_length = min(seg_a_len + seg_b_len, seq_len)
        seg_ids = [1] * min(seg_a_len, seq_len) 
        if has_text_b:
            seg_ids += [2] * min(seg_b_len, seq_len - len(seg_ids))
        
        # 统一填充到seq_len
        seg_ids += [0] * (seq_len - len(seg_ids))
        seg_buffer[i] = seg_ids
        
        # 标签处理
        label = int(parts[label_idx])
        if args.soft_targets and "logits" in columns:
            logits = list(map(float, parts[columns["logits"]].split()))
            batch.append((token_buffer[i], label, seg_buffer[i], logits))
        else:
            batch.append((token_buffer[i], label, seg_buffer[i]))
    
    return batch


def try_load_cache(path, args):
    """尝试加载缓存数据"""
    cache_key = generate_cache_key(path, args)
    cache_path = f"{path}.cache_{cache_key}"
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            metadata = pickle.load(f)
            if metadata['version'] == CACHE_VERSION and \
               metadata['seq_length'] == args.seq_length and \
               metadata['tokenizer'] == args.tokenizer.name_or_path:
                return metadata['data']
    return None

def save_cache(data, path, args):
    """保存缓存数据"""
    cache_key = generate_cache_key(path, args)
    cache_path = f"{path}.cache_{cache_key}"
    
    metadata = {
        'version': CACHE_VERSION,
        'seq_length': args.seq_length,
        'tokenizer': args.tokenizer.name_or_path,
        'data': data
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

def generate_cache_key(path, args):
    """生成缓存唯一标识"""
    file_hash = hashlib.md5(open(path, 'rb').read()).hexdigest()[:8]
    config_hash = f"{args.seq_length}_{args.tokenizer.name_or_path}"
    return f"{file_hash}_{hashlib.md5(config_hash.encode()).hexdigest()[:6]}"

def read_batch(mm, batch_size):
    """从内存映射中读取一个批次"""
    lines = []
    total_size = 0
    
    while total_size < batch_size * 1024:  # 以KB为单位控制批次
        pos = mm.tell()
        line = mm.readline()
        if not line:
            break
        lines.append(line.decode())
        total_size += len(line)
        
    return lines