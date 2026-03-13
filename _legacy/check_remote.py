import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('177.53.210.198', username='zzgabriel012', password='123')

stdin, stdout, stderr = client.exec_command('grep -r "MonsterCore V3" /home/zzgabriel012/aiworldmachinev13/')
print("STDOUT:", stdout.read().decode('utf-8'))
print("STDERR:", stderr.read().decode('utf-8'))

stdin, stdout, stderr = client.exec_command('ls -l /home/zzgabriel012/aiworldmachinev13/world_generator.py /home/zzgabriel012/aiworldmachinev13/monster_core_kernels.cu')
print("STDOUT2:", stdout.read().decode('utf-8'))
