import sys
import os
import subprocess


def check_and_install_nsight_system():
    nsight_path = 'NsightSystems-linux-public-2024.7.1.84-3512561.run'
    if not os.path.exists(nsight_path):
        print("Nsight Systems not installed, begin to install")
        #download_url = "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_7/NsightSystems-linux-public-2024.7.1.84-3512561.run"
        download_url = "https://developer.download.nvidia.cn/assets/tools/secure/nsight-systems/2020_3/NVIDIA_Nsight_Systems_Linux_CLI_Only_2020.3.1.72.deb"
        download_command = ['wget', download_url] 
        try:
            subprocess.run(download_command, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error when downloading Nsight Systems: {e}")
    
    subprocess.run(['sudo', 'yum', 'install', 'perl-Env'], text=True, check=True)
    subprocess.run(['chmod', '+x', 'NsightSystems-linux-public-2024.7.1.84-3512561.run'], text=True, check=True)
    subprocess.run(['./NsightSystems-linux-public-2024.7.1.84-3512561.run'], text=True, check=True)
    subprocess.run(['export PATH="/home/admin/nsight-systems-2024.7.1/bin:$PATH"'], text=True, check=True)


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


if __name__ == "__main__":
    # check_and_install_nsight_system()
    vpf_model_name = sys.argv[1]
    profile(vpf_model_name)