from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from ruamel.yaml import YAML
from pathlib import Path
import time, os, shutil

class FileSystemWatchdog(FileSystemEventHandler):
    """
    Asset watchdog
    Will, when the docker container is created and accessed watch asset/robots for new assets.
    Should any assets exist that does not have configuration files then these will be provided
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
        """Initial check of all folders for config files"""
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