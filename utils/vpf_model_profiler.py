import sys
import oss2
import base64
import uuid
import os
from time import localtime, strftime
import subprocess


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


def check_and_install_nsight_system():
    nsight_path = 'NsightSystems-linux-public-2024.7.1.84-3512561.run'
    if not os.path.exists(nsight_path):
        print("Nsight Systems not installed, begin to install")
        download_url = "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_7/NsightSystems-linux-public-2024.7.1.84-3512561.run"
        download_command = ['wget', download_url] 
        try:
            subprocess.run(download_command, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error when downloading Nsight Systems: {e}")
    
    subprocess.run(['sudo', 'yum', 'install', 'perl-Env'], text=True, check=True)
    subprocess.run(['chmod', '+x', 'NsightSystems-linux-public-2024.7.1.84-3512561.run'], text=True, check=True)
    subprocess.run(['./NsightSystems-linux-public-2024.7.1.84-3512561.run'], text=True, check=True)
    subprocess.run(['export PATH="/home/admin/nsight-systems-2024.7.1/bin:$PATH"'], text=True, check=True)

def create_file(real_model_name):
    file_content = '''
        from vpf_model_profiler import run_model
        
        if __name__ == "__main__":
            run_model()
    '''
    
    file_name = 'profile' + real_model_name,
    with open(file_name, 'w') as file:
        file.write(file_content)
    return file_name


def profile(vpf_model_name):
    _, _, real_model_name = vpf_model_name.partition("_")
    prefix, _, suffix = real_model_name.partition(".") 
    output_file = prefix + ".qdrep"
    run_model_name = 'run_' + prefix + '.py'
    
    print("Begin to profile")
    run_command = [
       'nsys',
       'profile',
       '-t', 'cuda,nvtx,osrt,cudnn,cublas',
       '-o', output_file,
       '-w', 'true',
       #'--cudabacktrace', 'true'
       'python3', run_model_name
   ]
    subprocess.run(run_command, text=True)
    print("Profile finished")
    save_output(output_file)


if __name__ == "__main__":
    # check_and_install_nsight_system()
    vpf_model_name = sys.argv[1]
    profile(vpf_model_name)