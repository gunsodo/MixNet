from mixnet.run_config import MixNetConfig
from mixnet.utils import load_yaml, get_yaml_path


if __name__ == "__main__":
    path = get_yaml_path()
    run_config = MixNetConfig(**load_yaml(path))
