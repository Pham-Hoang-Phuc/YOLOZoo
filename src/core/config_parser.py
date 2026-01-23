# src/core/config_parser.py
import yaml
import os
import sys

def merge_dicts(base, override):
    """
    Hàm đệ quy để gộp 2 dictionary.
    Override sẽ ghi đè lên Base nếu trùng key.
    """
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base

def load_config(cfg_path):
    """
    Load file config yaml có hỗ trợ kế thừa (base).
    """
    # Kiểm tra file có tồn tại không
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Nếu có danh sách 'base', load các file con trước
    if 'base' in cfg:
        base_cfg = {}
        # base có thể là 1 string hoặc 1 list
        bases = cfg['base'] if isinstance(cfg['base'], list) else [cfg['base']]
        
        for base_file in bases:
            # Xử lý đường dẫn: Ưu tiên đường dẫn tính từ root project
            # Nếu path bắt đầu bằng ./ hoặc ../ thì cần xử lý khéo hơn, 
            # nhưng ở đây ta giả định chạy từ root project.
            if not os.path.exists(base_file):
                print(f"Warning: Base file {base_file} not found relative to current dir.")
                
            # Load đệ quy (base của base)
            b_cfg = load_config(base_file)
            base_cfg = merge_dicts(base_cfg, b_cfg)
        
        # Gộp config hiện tại đè lên base config
        final_cfg = merge_dicts(base_cfg, cfg)
        
        # Xóa key base để output sạch đẹp
        if 'base' in final_cfg:
            del final_cfg['base']
            
        return final_cfg
    
    return cfg