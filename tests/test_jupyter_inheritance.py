import json
import pytest
from unittest import mock
from src.jupyter_inheritance import (
    _get_kernel_id,
    _get_kernel_connection_file_path,
    _dump_kernel_state,
    _wait_for_file,
    STORAGE_DIR,
)

_MODULE_PATH = "src.jupyter_inheritance"


def _assert_expected(fun, args, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            fun(*args)

    else:
        assert fun(*args) == expected


@pytest.mark.parametrize(
    "notebook_path, expected",
    [
        ("test/nb1.ipynb", "1"),
        ("test/nb2.ipynb", "2"),
        ("test/np3.ipynb", ValueError)
    ]
)
def test_get_kernel_id(notebook_path, expected):
    mock_server_metadata = [
        {
            "url": "test",
            "token": "test"
        },
    ]

    mock_sessions = [
        {
            "path": "test/nb1.ipynb",
            "kernel": {"id": "1"},
        },
        {
            "path": "test/nb2.ipynb",
            "kernel": {"id": "2"},
        }
    ]

    mock_response = mock.Mock(content=json.dumps(mock_sessions))

    patch_list_running_servers = mock.patch(
        f"{_MODULE_PATH}.list_running_servers",
        mock.Mock(return_value=mock_server_metadata)
    )

    patch_get = mock.patch("requests.get", mock.Mock(return_value=mock_response))

    with patch_get, patch_list_running_servers:
        _assert_expected(_get_kernel_id, [notebook_path], expected)


@pytest.mark.parametrize(
    "kernel_id, expected",
    [
        ("1", "test-dir/kernel-1.json"),
        ("2", "test-dir/kernel-2.json"),
        ("4", ValueError)
    ]
)
def test_get_kernel_connection_file_path(kernel_id, expected):

    patch_jupyter_runtime_dir = mock.patch(
        f"{_MODULE_PATH}.jupyter_runtime_dir", lambda: "test-dir"
    )

    patch_listdir = mock.patch(
        "os.listdir", lambda _: ["kernel-1.json", "kernel-2.json"]
    )

    with patch_jupyter_runtime_dir, patch_listdir:
        _assert_expected(
            _get_kernel_connection_file_path, [kernel_id], expected
        )


def test_dump_kernel_state():

    mock_client = mock.Mock()
    with mock.patch(
        f"{_MODULE_PATH}.BlockingKernelClient",
        mock.Mock(return_value=mock_client)
    ):
        _ = _dump_kernel_state("test.test", "123")

    expected_code = f"""
        import dill
        dill.dump_session("{STORAGE_DIR}/123.sesh")
    """
    mock_client.execute.assert_called_with(expected_code)


def test_wait_for_file_exists():
    with mock.patch("os.path.exists", lambda _: True):
        _wait_for_file("test_path", 0.1)


def test_wait_for_file_not_exists():
    with mock.patch("os.path.exists", lambda _: False):
        with pytest.raises(RuntimeError):
            _wait_for_file("test_path", 0.1)
