import os
import subprocess

def check_docker_running():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_data_path():
    """Get correct data path for Docker vs local"""
    if os.path.exists('/app/data'):
        return '/app/data'
    return './data'

def get_model_path():
    """Get correct model path for Docker vs local"""
    if os.path.exists('/app/models'):
        return '/app/models'
    return './models'

def get_db_path():
    """Get correct DB path for Docker vs local"""
    if os.path.exists('/app/sql'):
        return '/app/sql/credit_risk.db'
    return './sql/credit_risk.db'