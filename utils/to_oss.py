from time import localtime, strftime
import uuid
import oss2
import base64
import sys


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


if __name__ == "__main__":
    file_name = sys.argv[1]
    save_output(file_name)