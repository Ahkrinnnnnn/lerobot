import sys
import os
import ctypes


def load_CrpRobotPy(third_party_root=None):
    """
    third_party_root: third_party 文件夹的绝对路径
    将 third_party/CrpRobotPy 加入 sys.path，更新 LD_LIBRARY_PATH，并尝试预加载 helper .so。
    返回 (crp_pkg_path, lib_dir)
    """
    if third_party_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        third_party_root = os.path.join(repo_root, "third_party")

    crp_pkg_path = os.path.join(third_party_root, "CrpRobotPy")
    if crp_pkg_path not in sys.path:
        sys.path.insert(0, crp_pkg_path)

    lib_dir = crp_pkg_path
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{old_ld}" if old_ld else lib_dir

    for name in ("libRobotService.so", "CrpRobotPy.so"):
        helper_so = os.path.join(lib_dir, name)
        try:
            if os.path.exists(helper_so):
                ctypes.CDLL(helper_so)
        except OSError:
            pass

    return crp_pkg_path, lib_dir
