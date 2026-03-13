import paramiko
import sys

print("Upload started", flush=True)

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('177.53.210.198', username='zzgabriel012', password='123', timeout=10)
    print("Connected", flush=True)

    sftp = client.open_sftp()
    
    localpath = 'c:/Users/Pichau/Downloads/AI_World_Engine/world_generator.py'
    remotepath = '/home/zzgabriel012/aiworldmachinev13/world_generator.py'
    
    sftp.put(localpath, remotepath)
    
    sftp.close()
    client.close()
    print("Upload concluded successfully", flush=True)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
