import paramiko
from scp import SCPClient
import os

host = "192.168.50.218"
user = "gpuuser"
password = "fMReDquSrYwN"

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=10)

    scp = SCPClient(client.get_transport())
    
    # Sync entire app directory
    scp.put("app", remote_path="~/star-router/", recursive=True)
    
    # Restart the server securely loading .env
    stdin, stdout, stderr = client.exec_command("cd star-router && pkill -f 'uvicorn app.main:app' || true")
    stdout.channel.recv_exit_status()
    
    # Install new dependencies explicitly
    stdin, stdout, stderr = client.exec_command("cd star-router && /usr/bin/python3 -m pip install openai anthropic")
    stdout.channel.recv_exit_status()
    
    # Ensure export from .env works
    stdin, stdout, stderr = client.exec_command("cd star-router && set -a && [ -f .env ] && source .env && set +a && nohup /usr/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 > router.log 2>&1 & echo $! > router.pid")
    stdout.channel.recv_exit_status()
    
    scp.close()
    client.close()
    print("Agent Dispatch synced and remote server restarted!")
except Exception as e:
    print(f"Error: {e}")
