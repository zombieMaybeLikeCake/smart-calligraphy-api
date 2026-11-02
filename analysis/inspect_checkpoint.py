"""唯讀：不載入 torch，用受限的 Unpickler 只讀出 *.pth 裡每個 tensor 的 key / shape / dtype。
用法： python analysis/inspect_checkpoint.py <pth 檔路徑>...
只允許白名單內的 pickle global，其餘一律拒絕（檔案是不信任資料）。"""
import collections, pickle, sys, zipfile, os


class _Storage:
    def __init__(self, name): self.name = name


class _Tensor:
    def __init__(self, shape, dtype): self.shape, self.dtype = tuple(shape), dtype


def _rebuild_tensor_v2(storage, offset, size, stride, *a, **k):
    return _Tensor(size, storage.name)


class _Unp(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel)
        return _Storage(pid[1] if isinstance(pid[1], str) else getattr(pid[1], "__name__", str(pid[1])))


def inspect(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    sd = _Unp(z.open(pkl)).load()
    print(f"== {path}  ({os.path.getsize(path)} bytes, {len(sd)} entries)")
    total = 0
    for k, v in sd.items():
        n = 1
        for s in v.shape: n *= s
        total += n
        print(f"  {k:55s} {str(v.shape):22s} {v.dtype}")
    print(f"  total params (incl. BN buffers): {total}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
