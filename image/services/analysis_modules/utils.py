from pathlib import Path


def list_file_paths(dir_path: str | Path) -> list[Path]:
    return sorted(
        [
            str(path)
            for path in Path(dir_path).rglob("*")
            if path.is_file()
            and path.suffix in [".png", ".jpg", ".jpeg", ".bmp", "webp"]
        ]
    )
