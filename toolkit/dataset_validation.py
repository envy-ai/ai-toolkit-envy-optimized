"""Actionable, aggregate reports for datasets that require paired images."""
import os
from PIL import Image, ImageOps


class DatasetValidationError(ValueError):
    """Expected input failure, displayed without a Python traceback."""
    sqlite_db_path = './aitk_db.db'


def paired_image_issues(banks, dimension_folders=None):
    """banks maps folder paths to filename stems to lists of matching files.

    Edit references can require matching names without requiring target sizes.
    """
    stems = set().union(*(set(bank) for bank in banks.values()))
    issues = []
    missing = []
    for folder, bank in banks.items():
        absent = sorted(stems - set(bank))
        lines = []
        for stem in absent:
            names = sorted({os.path.basename(path) for other in banks.values()
                            for path in other.get(stem, [])})
            lines.append(f'    {stem} (present elsewhere as: {", ".join(names)})')
        missing.append(f'  {folder}:\n' + ('\n'.join(lines) if lines else '    None'))
        if not bank:
            issues.append(f'Folder has no usable images: {folder}')
        for stem, paths in sorted(bank.items()):
            if len(paths) > 1:
                issues.append(f'ambiguous filename stem {stem}:\n' + '\n'.join(f'  {p}' for p in paths))
    if any(stems - set(bank) for bank in banks.values()):
        issues.insert(0, 'Missing matching filename stems by directory:\n' + '\n'.join(missing))

    size_folders = set(banks) if dimension_folders is None else set(dimension_folders)
    for stem in sorted(stems):
        sizes = []
        for folder, bank in banks.items():
            for path in bank.get(stem, []):
                try:
                    with Image.open(path) as image:
                        size = ImageOps.exif_transpose(image).size
                    if folder in size_folders:
                        sizes.append((path, size))
                except (OSError, ValueError) as error:
                    issues.append(f'Cannot read image: {path}: {error}')
        if len({size for _, size in sizes}) > 1:
            issues.append(f'Resolution mismatch (different dimensions), {stem}:\n' + '\n'.join(
                f'  {path}: {size[0]} x {size[1]}' for path, size in sizes))
    return issues


def raise_dataset_issues(issues):
    if issues:
        raise DatasetValidationError('Dataset validation failed. Training stopped before loading the model.\n\n'
                                     + '\n\n'.join(issues)
                                     + '\n\nCorrect the listed files, then restart the job.')
