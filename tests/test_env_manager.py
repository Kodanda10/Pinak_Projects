import pytest
import time
import os
import json
from pinak.env_manager.manager import EnvManager

@pytest.fixture
def env_manager(tmp_path):
    # Create a dummy config for testing
    config_path = tmp_path / "test_project.json"
    config_data = {
        "scripts": {
            "test_sleep": "sleep 1",
            "bad_command": "this_command_does_not_exist",
            "log_printer": [
                "python3", "-u", "-c",
                "import sys; print('Hello from stdout'); sys.stderr.write('Error from stderr\\n'); sys.exit(0)"
            ]
        }
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    
    em = EnvManager(str(config_path)) # Pass string path
    yield em
    
    # Teardown: stop all processes and remove dummy config
    em.stop_all()
    # tmp_path handles cleanup of the file itself

def test_start_and_stop_process(env_manager):
    """Tests the basic functionality of starting and stopping a named process."""
    script_name = "test_sleep"
    
    # 1. Start the process
    env_manager.run(script_name)
    time.sleep(0.5) # Give it a moment to start
    
    # 2. Check its status
    status = env_manager.get_status(script_name)
    assert status['status'] == 'running'
    assert status['pid'] is not None
    
    # 3. Stop the process
    env_manager.stop(script_name)
    time.sleep(0.5) # Give it a moment to stop
    
    # 4. Verify it has stopped
    status = env_manager.get_status(script_name)
    assert status['status'] == 'terminated'


def test_log_capture_and_error_handling(env_manager):
    """Tests log capture and error reporting for failed processes."""
    
    # Test Log Capture
    log_script_name = "log_printer"
    env_manager.run(log_script_name)
    time.sleep(1) # Give it time to print and exit
    
    logs = env_manager.get_logs(log_script_name)
    assert len(logs) >= 2
    assert any("Hello from stdout" in log for log in logs)
    assert any("Error from stderr" in log for log in logs)
    
    status = env_manager.get_status(log_script_name)
    assert status['status'] in ('stopped','terminated') # It exits quickly

    # Test Error Handling (Process Failure)
    bad_script_name = "bad_command"
    env_manager.run(bad_script_name)
    time.sleep(0.5) # Give it a moment to try and fail
    
    status = env_manager.get_status(bad_script_name)
    assert status['status'].startswith('failed')
    # We may not expose return_code consistently; ensure not running
    assert status['status'].startswith('failed') or status['status'] in ('stopped','terminated')
