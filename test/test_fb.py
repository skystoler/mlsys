def parse_scheduler(scheduler):
    fbconfig = []
    if ";" in scheduler:
        config_list = scheduler.split(";")
    else:
        config_list = [scheduler]
    for config in config_list:
        if not config or "," not in config:
            continue
        fbconfig.append([eval(n) for n in config.split(",")])
    print(fbconfig)
    using_validation = True
    config_length = len(fbconfig)
    for idx in range(config_length):
        cur_config = fbconfig[idx]
        next_config = fbconfig[idx + 1] if idx < config_length - 1 else None
        residual_diff_threshold, start, end, max_consecutive_cache_hits = cur_config
        
        print(residual_diff_threshold, start, end, max_consecutive_cache_hits)
        if max_consecutive_cache_hits < 0 and start <= 0 and end >= 1:
            using_validation = False
        if residual_diff_threshold <= 0.0 or max_consecutive_cache_hits == 0 or len(cur_config) != 4:
            print("no")
        if next_config is not None and end > next_config[1]:
            print("no", end, next_config[1])
        print(using_validation)
        
    create_func(residual_diff_threshold = [config[1] for config in fbconfig])

def create_func(residual_diff_threshold):
    if isinstance(residual_diff_threshold, list):
        print(residual_diff_threshold[0])
    else:
        print(residual_diff_threshold)
        
#parse_scheduler(scheduler="0.12,0.00,1.00,5")
parse_scheduler(scheduler="0.12,0.0,0.9,5;0.12,0.9,1.0,5")