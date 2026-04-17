import paramiko
host = "192.168.50.218"
user = "gpuuser"
password = "fMReDquSrYwN"

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=10)

    stdin, stdout, stderr = client.exec_command("cd star-router && tail -n 50 router.log")
    print(stdout.read().decode())
    
    stdin, stdout, stderr = client.exec_command("ps aux | grep uvicorn")
    print("PS AUX:\n", stdout.read().decode())

    client.close()
except Exception as e:
    print(f"Error: {e}")
