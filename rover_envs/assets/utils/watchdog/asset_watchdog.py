from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from ruamel.yaml import YAML
from pathlib import Path
import time, os, shutil

class FileSystemWatchdog(FileSystemEventHandler):
    """
    FileSystemWatchdog monitors the 'robots' asset directory and automatically injects default
    configuration files into any newly created robot folders.

    This is designed for use within a containerized simulation environment, where asset directories
    (such as those representing individual rover models) may be added dynamically at runtime or
    through host-system bindings (e.g. bind mounts in Docker).

    Functionality:
    - At startup, it iterates over all existing folders in `robots/`, and if a `configs/` subfolder
      is missing, it copies in default YAML configurations from a central template directory.
    - During runtime, it observes for newly created folders, and upon creation, does the same config
      injection automatically.
    - Ownership (UID/GID) of the copied files is preserved from the template source to avoid
      permission issues inside the container.

    Example watched structure:
        rover_envs/
        └── assets/
            └── robots/
                ├── aau_rover/
                │   └── configs/  ← auto-created if missing
                └── leo_rover/
                    └── configs/

    Attributes:
        _CONFIGS_DIR (Path): Path to the default configuration templates.
        paths_dir (Path): The base directory being monitored (i.e. assets/robots).
        paths_dirs (list): Currently tracked robot folders.
        observer (PollingObserver): Filesystem observer instance using polling backend.
    """
    _CONFIGS_DIR = Path(__file__).resolve().parents[0] / "configs"

    def __init__(self):
        self.paths_dirs = []

        # Init directories for robot assets
        self.paths_dir = Path(__file__).resolve().parents[2] / "robots"
        for folder in self.paths_dir.iterdir():
            if folder.is_dir():
                self.paths_dirs.append(folder)

        # Init template access info
        self._CONFIG_DIR_STAT = os.stat(self._CONFIGS_DIR)
        self._CONFIG_FILES_STAT = os.stat(self._CONFIGS_DIR / "robot_default.yaml")
        self.yaml = YAML()
        self.init_process_folders()

        # Setup observer
        print(f"[Watchdog] Watching path: {self.paths_dir}")
        self.observer = Observer()
        self.observer.schedule(self, self.paths_dir, recursive=False)
        self.observer.start()

    def on_created(self, event):
        """
                Callback function triggered when a new directory is created in the observed path.

                When a new robot asset folder is added, this method checks whether a `configs/` subfolder exists.
                If not, it copies the default configuration templates from `_CONFIGS_DIR` into the new folder.
                It also replicates the UID and GID from the template directory to the copied files to avoid
                permission issues in Docker or remote environments.

                Args:
                    event (DirCreatedEvent): Filesystem event containing the path of the created directory.
        """
        if not event.is_directory:
            return

        print(f"[Watchdog] Directory created: {event.src_path}")
        config_path = Path(event.src_path) / "configs"
        if not config_path.exists():
            try:
                shutil.copytree(self._CONFIGS_DIR, config_path, dirs_exist_ok=True)
                os.chown(config_path, self._CONFIG_DIR_STAT.st_uid, self._CONFIG_DIR_STAT.st_gid)
                for file in config_path.rglob("*"):
                    os.chown(file, self._CONFIG_FILES_STAT.st_uid, self._CONFIG_FILES_STAT.st_gid)
                print(f"[Watchdog] Configs copied to: {config_path}")
            except Exception as e:
                print(f"[Watchdog] Error copying configs: {e}")

        self.paths_dirs.append(event.src_path)

    def init_process_folders(self):
        """
        Performs an initial scan of all existing robot asset folders inside the monitored directory.

        For each folder:
        - If a `configs/` subfolder is missing, the default configuration files are copied into place.
        - Ownership is preserved by replicating the UID/GID from the template source.

        This ensures that even robot assets added before the watchdog was started will receive
        the required configuration structure.
        """
        for folder in self.paths_dir.iterdir():
            if folder.is_dir() and folder.name not in {"__pycache__", ".vscode"}:
                config_path = folder / "configs"
                if not config_path.exists():
                    shutil.copytree(self._CONFIGS_DIR, config_path, dirs_exist_ok=True)
                    os.chown(config_path, self._CONFIG_DIR_STAT.st_uid, self._CONFIG_DIR_STAT.st_gid)
                    for file in config_path.rglob("*"):
                        os.chown(file, self._CONFIG_FILES_STAT.st_uid, self._CONFIG_FILES_STAT.st_gid)

if __name__ == '__main__':
    print("[Watchdog] Script started.")
    Watchdog = FileSystemWatchdog()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Watchdog] Shutting down...")
        Watchdog.observer.stop()
    Watchdog.observer.join()