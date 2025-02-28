import torch
import functools
from typing import Optional
from time import localtime, strftime
import uuid
import oss2
import base64
import pickle


def get_encoded_credentials():
    """解码访问密钥"""
    encoded_access_key_id = b'TFRBSTV0UnQ2Nlc3ekdzcWtNWXBpYk10'
    encoded_access_key_secret = b'aXpPTTRQYVQ5OWNOOFgxdmdXMnN5d0FnWXpCMGFs'

    return (
        base64.b64decode(encoded_access_key_id).decode('utf-8'),
        base64.b64decode(encoded_access_key_secret).decode('utf-8')
    )


def upload_to_oss(file_content, filename):
    """上传文件到阿里云OSS存储"""
    access_key_id, access_key_secret = get_encoded_credentials()

    endpoint = "http://oss-cn-beijing.aliyuncs.com"
    bucket_name = "sm-compute-public-data"
    root_directory = "zty"

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    current_time = strftime('%Y-%m-%d-%H:%M:%S', localtime())
    object_path = "{}/{}/{}/{}/{}".format(
        root_directory,
        "profile",
        current_time,
        str(uuid.uuid4()),
        filename
    )
    bucket.put_object(object_path, file_content)
    print(f"Profile output URL: http://sm-compute-public-data.oss-cn-beijing.aliyuncs.com/{object_path}")


def save_output(output_filename):
    """保存输出文件至指定位置"""
    try:
        with open(output_filename, "rb") as file:
            file_content = file.read()
        
        upload_to_oss(file_content, output_filename)
    except Exception as e:
        print(f"Error saving the output: {e}")


class MemorySnapshotDecorator:
    """显存分配追踪装饰器类"""
    def __init__(self, snapshot_file: str = "memory_snapshot.pickle"):
        self.snapshot_file = snapshot_file
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查CUDA可用性
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            torch.cuda.empty_cache()
            
            try:
                torch.cuda.memory._record_memory_history(
                    # trace_alloc_max_entries=100000,
                    # trace_free_max_entries=100000
                )
                
                # 执行目标函数
                result = func(*args, **kwargs)
                
                return result
            finally:
                # 无论是否异常都执行以下操作
                try:
                    # 生成内存快照
                    torch.cuda.memory._dump_snapshot(self.snapshot_file)
                    print(f"Memory snapshot saved to {self.snapshot_file}")
                finally:
                    # 停止记录内存历史
                    torch.cuda.memory._record_memory_history(enabled=None)
                    torch.cuda.empty_cache()

        return wrapper


def memory_snapshot(snapshot_file: Optional[str] = None):
    return MemorySnapshotDecorator(snapshot_file=snapshot_file)


@memory_snapshot(snapshot_file="snapshot.pickle")
def run_model_inference():
    model = torch.nn.Linear(100, 100).cuda()
    inputs = torch.randn(32, 100).cuda()
    
    for _ in range(10):
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        
    del model, inputs, outputs


def analyze_snapshot(snapshot_path):
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)
        
    # 分析分配记录
    for seg in snapshot['segments']:
        print(seg.keys())
        print(f"分配 {seg['allocated_size']} bytes 在设备 {seg['device']}")

if __name__ == "__main__":
    run_model_inference()
    # analyze_snapshot("snapshot.pickle")
    save_output(output_filename="snapshot.pickle")