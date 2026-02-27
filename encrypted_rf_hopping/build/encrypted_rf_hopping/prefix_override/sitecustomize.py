import sys
if sys.prefix == '/home/irman/micromamba/envs/ros2_env':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/irman/Combat-UGV/encrypted_rf_hopping/install/encrypted_rf_hopping'
