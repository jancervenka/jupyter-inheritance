import os
import json
import dill
import time
import requests
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_server.serverapp import list_running_servers
from jupyter_client.blocking import BlockingKernelClient

STORAGE_DIR = os.path.join(os.path.expanduser("~"), ".jupyter-inheritance")
DEFAULT_TIMEOUT = 60.


def _get_kernel_id(notebook_path: str) -> str:
    """
    Finds the `kernel_id` for a given `notebook_path` via Jupyter Server API.
    """

    # there can be more than one server but that's ok
    server_metadata = list(list_running_servers())[0]
    server_url = server_metadata["url"]
    server_token = server_metadata["token"]

    sessions_url = f"{server_url}api/sessions?token={server_token}"
    response = requests.get(sessions_url)
    response.raise_for_status()
    all_sessions = json.loads(response.content)

    session = filter(lambda x: x["path"] == notebook_path, all_sessions)
    try:
        return list(session)[0]["kernel"]["id"]
    except (IndexError, KeyError):
        raise ValueError(f"No running notebook '{notebook_path}' found.")


def _get_kernel_connection_file_path(kernel_id: str) -> str:
    """
    Finds the path to the connection file in the runtime directory
    for a given `kernel_id`.
    """

    all_connection_files = os.listdir(jupyter_runtime_dir())
    connection_file = filter(
        lambda x: x == f"kernel-{kernel_id}.json", all_connection_files
    )

    try:
        return os.path.join(jupyter_runtime_dir(), list(connection_file)[0])
    except IndexError:
        raise ValueError(f"No connection file for '{kernel_id}' found.")


def _dump_kernel_state(connection_file_path: str, kernel_id: str) -> str:
    """
    Connects to a kernel defined by the `connection_file_path`
    and dumps its state to the `storage_file_path`.
    """

    storage_file_path = os.path.join(STORAGE_DIR, f"{kernel_id}.sesh")

    code = f"""
        import dill
        dill.dump_session("{storage_file_path}")
    """

    client = BlockingKernelClient(connection_file=connection_file_path)
    client.load_connection_file()
    client.start_channels()
    _ = client.execute(code)
    # calling `client.get_shell_msg` should block this process until
    # the serialization is complete to prevent loading an incomplete file
    _ = client.get_shell_msg()
    return storage_file_path


def _wait_for_file(storage_file_path: str, timeout: float):
    """
    This can be removed because we are blocking with `client.get_shell_msg()`.
    """

    start = time.time()
    while (time.time() - start) < timeout:
        if os.path.exists(storage_file_path):
            return

    raise RuntimeError("Timeout while waiting for serialized kernel.")


def inherit_from(notebook_path: str, timeout: float = DEFAULT_TIMEOUT) -> None:
    """
    Inherits from a notebook specified by the `notebook_path`. `timeout`
    defines how long to wait (in seconds) for the notebook kernel to be serialized.
    """

    os.makedirs(STORAGE_DIR, exist_ok=True)
    kernel_id = _get_kernel_id(notebook_path)
    connection_file_path = _get_kernel_connection_file_path(kernel_id)
    storage_file_path = _dump_kernel_state(connection_file_path, kernel_id)
    _wait_for_file(storage_file_path, timeout)
    dill.load_session(storage_file_path)
    if os.path.exists(storage_file_path):
        os.remove(storage_file_path)
