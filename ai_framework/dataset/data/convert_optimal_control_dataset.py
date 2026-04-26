"""Convert the dataset into real/imaginary STFT file pairs.

The canonical dataset schema now stores two STFT component columns:
- ``stft_data_real``: filename for the real component array.
- ``stft_data_imaginary``: filename for the imaginary component array.

This script can start from either:
- the current file-based CSV with ``STFT_Complex_File`` references, or
- the inline CSV export with ``STFT_Complex`` dict payloads.

The output files are regenerated as float32 arrays in the existing data folders
so the loader can reconstruct a complex tensor on demand.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from shutil import rmtree

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = DATA_DIR / "optimal_control_dataset (1).csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "optimal_control_dataset.csv"
STFT_DATA_DIR = DATA_DIR / "stft_data"
STFT_COMPLEX_DIR = DATA_DIR / "stft_complex"

SOURCE_INLINE_DATA_COLUMN = "STFT_Data"
SOURCE_INLINE_COMPLEX_COLUMN = "STFT_Complex"
SOURCE_FILE_COLUMN = "STFT_Complex_File"
OUTPUT_REAL_COLUMN = "stft_data_real"
OUTPUT_IMAGINARY_COLUMN = "stft_data_imaginary"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help="Path to the inline-STFT CSV export.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the regenerated filename-based CSV.",
    )
    parser.add_argument(
        "--keep-source-csv",
        action="store_true",
        help="Keep the inline source CSV after conversion.",
    )
    return parser.parse_args()


def _reset_directory(path: Path) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _replace_directory(src: Path, dst: Path) -> None:
    if dst.exists():
        rmtree(dst)
    src.rename(dst)


def _parse_flat_float_list(payload: str, field_name: str) -> np.ndarray:
    value = ast.literal_eval(payload)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1).astype(np.float32)
    return array


def _parse_complex_payload(payload: str) -> np.ndarray:
    value = ast.literal_eval(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{SOURCE_INLINE_COMPLEX_COLUMN} must be a dict-like payload")

    try:
        real = np.asarray(value["real"], dtype=np.float32).reshape(-1)
        imag = np.asarray(value["imag"], dtype=np.float32).reshape(-1)
        shape_raw = value["shape"]
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"{SOURCE_INLINE_COMPLEX_COLUMN} is missing required key: {missing}") from exc

    shape = tuple(int(dim) for dim in shape_raw)
    if len(shape) != 2:
        raise ValueError(f"{SOURCE_INLINE_COMPLEX_COLUMN} shape must be 2D, got {shape!r}")

    expected_size = int(np.prod(shape))
    if real.size != imag.size or real.size != expected_size:
        raise ValueError(
            f"{SOURCE_INLINE_COMPLEX_COLUMN} payload size mismatch: real={real.size}, "
            f"imag={imag.size}, shape={shape}, expected={expected_size}"
        )

    return (real + 1j * imag).astype(np.complex64).reshape(shape)


def _load_complex_from_file(data_root: Path, filename: str) -> np.ndarray:
    stft_path = data_root / "stft_complex" / filename
    if not stft_path.exists():
        raise FileNotFoundError(f"STFT file not found: {stft_path}")
    stft = np.load(stft_path)
    if not np.iscomplexobj(stft):
        raise ValueError(f"Expected complex STFT file, got {stft.dtype} from {stft_path}")
    return np.asarray(stft, dtype=np.complex64)


def _sample_filename(prefix: str, index: int) -> str:
    width = max(4, len(str(index)))
    return f"{prefix}_{index:0{width}d}.npy"


def _load_rows(source_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    csv.field_size_limit(sys.maxsize)
    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {source_csv}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def _write_outputs(rows: list[dict[str, str]], fieldnames: list[str], output_csv: Path) -> None:
    has_inline_payload = SOURCE_INLINE_COMPLEX_COLUMN in fieldnames
    has_file_column = SOURCE_FILE_COLUMN in fieldnames
    if not has_inline_payload and not has_file_column:
        raise ValueError(
            f"Expected {SOURCE_INLINE_COMPLEX_COLUMN!r} or {SOURCE_FILE_COLUMN!r} in source CSV"
        )

    excluded = {SOURCE_INLINE_DATA_COLUMN, SOURCE_INLINE_COMPLEX_COLUMN, SOURCE_FILE_COLUMN}
    metadata_fields = [name for name in fieldnames if name not in excluded]
    output_fields = metadata_fields + [OUTPUT_REAL_COLUMN, OUTPUT_IMAGINARY_COLUMN]

    tmp_data_dir = STFT_DATA_DIR.parent / (STFT_DATA_DIR.name + "_tmp")
    tmp_complex_dir = STFT_COMPLEX_DIR.parent / (STFT_COMPLEX_DIR.name + "_tmp")
    tmp_output_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")

    _reset_directory(tmp_data_dir)
    _reset_directory(tmp_complex_dir)

    with tmp_output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for index, row in enumerate(rows):
            if has_inline_payload:
                complex_payload = row.get(SOURCE_INLINE_COMPLEX_COLUMN)
                if complex_payload is None:
                    raise ValueError(f"Row {index} is missing {SOURCE_INLINE_COMPLEX_COLUMN}")
                stft_complex = _parse_complex_payload(complex_payload)
            else:
                complex_filename = row.get(SOURCE_FILE_COLUMN)
                if complex_filename is None:
                    raise ValueError(f"Row {index} is missing {SOURCE_FILE_COLUMN}")
                stft_complex = _load_complex_from_file(DATA_DIR, complex_filename)

            stft_real = np.asarray(stft_complex.real, dtype=np.float32)
            stft_imag = np.asarray(stft_complex.imag, dtype=np.float32)

            real_filename = _sample_filename("stft_data_real", index)
            imaginary_filename = _sample_filename("stft_data_imaginary", index)

            np.save(tmp_data_dir / real_filename, stft_real, allow_pickle=False)
            np.save(tmp_complex_dir / imaginary_filename, stft_imag, allow_pickle=False)

            output_row = {name: row.get(name, "") for name in metadata_fields}
            output_row[OUTPUT_REAL_COLUMN] = real_filename
            output_row[OUTPUT_IMAGINARY_COLUMN] = imaginary_filename
            writer.writerow(output_row)

    _replace_directory(tmp_data_dir, STFT_DATA_DIR)
    _replace_directory(tmp_complex_dir, STFT_COMPLEX_DIR)
    tmp_output_csv.replace(output_csv)


def main() -> int:
    args = _parse_args()
    source_csv: Path = args.source_csv
    output_csv: Path = args.output_csv

    if not source_csv.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {source_csv}. "
            "Pass --source-csv to point to your inline export file."
        )

    rows, fieldnames = _load_rows(source_csv)
    if not rows:
        raise ValueError(f"Source CSV has no data rows: {source_csv}")

    _write_outputs(rows, fieldnames, output_csv)

    if (
        not args.keep_source_csv
        and source_csv.resolve() != output_csv.resolve()
        and source_csv.parent.resolve() == DATA_DIR.resolve()
    ):
        source_csv.unlink()

    print(f"Converted {len(rows)} rows from {source_csv.name} -> {output_csv.name}")
    print(f"Wrote {len(rows)} real files to {STFT_DATA_DIR.name}/ and {len(rows)} imaginary files to {STFT_COMPLEX_DIR.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())