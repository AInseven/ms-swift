"""
服务器运行脚本
从H800 copy lora 到 H20
"""
import os
try:
    import paramiko
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "scp"])
    import paramiko

from scp import SCPClient
# ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# cat ~/.ssh/id_rsa.pub
# === CONFIG ===
parent_dir = "/root/output/GRPO_MCP/v1-20250726-165033"
sub_dirs = [
    # "train_2025-07-14-18-28-12/checkpoint-350",
    # "train_2025-07-14-18-28-12/checkpoint-450",
    # "train_2025-07-14-18-28-12/checkpoint-480",
    # "train_2025-07-14-18-28-12/checkpoint-250",
    # "train_2025-07-14-18-28-12/checkpoint-200",
    # "train_2025-07-14-19-56-11/checkpoint-300",
    "checkpoint-70",
    "checkpoint-30",
    "checkpoint-60",
]

current_ssh_key=os.path.expanduser("~/.ssh/id_rsa")

files_to_copy = ["adapter_model.safetensors", "adapter_config.json"]


servers = {
    "L20-093": {
        "host": "connect.bjc1.seetacloud.com",
        "user": "root",
        "port": 55178,
    },
    "L20-098": {
        "host": "connect.bjc1.seetacloud.com",
        "user": "root",
        "port": 31289,
    },
    "北京A区 / 699机": {
        "host": "region-42.seetacloud.com",
        "user": "root",
        "port": 46807,
    },
    "北京A区 / 701机": {
        "host": "region-42.seetacloud.com",
        "user": "root",
        "port": 23669,
    },
}

remote_path = "/root/autodl-tmp/h800lora"


# === FUNCTION ===
def create_ssh_client(host, port, user):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        key_filename=current_ssh_key
    )
    return client


def ensure_remote_dir(ssh_client, remote_dir):
    cmd = f"mkdir -p {remote_dir}"
    ssh_client.exec_command(cmd)


def copy_files(ssh_client, local_path, remote_full_path):
    def progress(filename, size, sent):
        mb = lambda b: f"{b / 1024**2:.2f} MB"
        percent = sent / size * 100 if size != 0 else 100
        print(f"  ↳ {filename}: {mb(sent)} / {mb(size)} ({percent:.1f}%)", end='\r')


    with SCPClient(ssh_client.get_transport(), progress=progress) as scp:
        scp.put(local_path, remote_full_path)
    print()  # newline after progress


def connect_to(server_name: str):
    if server_name not in servers:
        raise ValueError(f"Unknown server: {server_name}")

    config = servers[server_name]
    ssh = create_ssh_client(
        config["host"],
        config["port"],
        config["user"]
    )
    return ssh

# === MAIN LOGIC ===
ssh = connect_to('北京A区 / 701机')

for sub in sub_dirs:
    subdir_path = os.path.join(parent_dir, sub)
    remote_subdir = os.path.join(remote_path, sub)

    # Create remote subdir
    ensure_remote_dir(ssh, remote_subdir)

    for filename in files_to_copy:
        file_path = os.path.join(subdir_path, filename)
        if os.path.exists(file_path):
            remote_file_path = os.path.join(remote_subdir, filename)
            progress = f"({sub_dirs.index(sub) * len(files_to_copy) + files_to_copy.index(filename) + 1}/{len(sub_dirs) * len(files_to_copy)})"
            print(f"Copying {progress}: {file_path} → {remote_file_path}")
            copy_files(ssh, file_path, remote_file_path)
        else:
            print(f"⚠️ Missing file: {file_path}")

ssh.close()

