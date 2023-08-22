import os

try:
    import soccer_twos
except ModuleNotFoundError:
    temp_envs_path = r'./temp/envs'
    if not os.path.exists(temp_envs_path):
        os.mkdir(temp_envs_path)
    from git import Repo
    if not os.path.exists(os.path.join(temp_envs_path, r'soccer_twos')):
        Repo.clone_from(r'https://github.com/s-sd/soccer-twos-env.git', os.path.join(temp_envs_path, r'soccer_twos'))
    import pip
    pip.main(['install', '-e', r'./temp/envs/soccer_twos/'])
    import soccer_twos
    
env = soccer_twos.make()
