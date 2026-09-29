from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from amesml import AmesSchemaError, DatasetDownloadError, load_ames, train_test_split_df
from amesml import data as data_module
from amesml.data import (
    DATASET_SHA256,
    FEATURE_COLUMNS,
    dataset_fingerprint,
    download_ames,
    read_ames_csv,
    validate_ames,
)


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")


def fake_get(content: bytes, status: int = 200) -> Any:
    def get(url: str, timeout: float) -> FakeResponse:
        return FakeResponse(content, status)

    return get


def test_feature_schema_has_79_columns() -> None:
    assert len(FEATURE_COLUMNS) == 79
    assert len(set(FEATURE_COLUMNS)) == 79


def test_na_means_missing_but_the_word_none_is_a_category(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("Id,MasVnrType,PoolQC,LotFrontage\n1,None,NA,65\n2,BrkFace,Gd,\n", "utf-8")
    frame = read_ames_csv(path)
    assert frame.loc[0, "MasVnrType"] == "None"
    assert pd.isna(frame.loc[0, "PoolQC"])
    assert pd.isna(frame.loc[1, "LotFrontage"])


def test_valid_frame_passes(ames_frame: pd.DataFrame) -> None:
    validate_ames(ames_frame)


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(AmesSchemaError, match="empty"):
        validate_ames(pd.DataFrame())


def test_missing_columns_are_listed(ames_frame: pd.DataFrame) -> None:
    frame = ames_frame.drop(columns=["GrLivArea", "Neighborhood"])
    with pytest.raises(AmesSchemaError, match="missing 2 column.*GrLivArea, Neighborhood"):
        validate_ames(frame)


def test_long_lists_of_missing_columns_are_shortened(ames_frame: pd.DataFrame) -> None:
    frame = ames_frame[["Id", "SalePrice"]]
    with pytest.raises(AmesSchemaError, match=r"missing 79 column.*\(\+71 more\)"):
        validate_ames(frame)


def test_non_numeric_values_are_reported_with_an_example(ames_frame: pd.DataFrame) -> None:
    frame = ames_frame.copy()
    frame["LotArea"] = frame["LotArea"].astype(object)
    frame.loc[3, "LotArea"] = "big"
    with pytest.raises(AmesSchemaError, match="'LotArea' must be numeric but contains 'big'"):
        validate_ames(frame)


@pytest.mark.parametrize(
    ("value", "message"), [(float("nan"), "missing value"), (0, "must be positive")]
)
def test_bad_target_values(ames_frame: pd.DataFrame, value: float, message: str) -> None:
    frame = ames_frame.copy()
    frame["SalePrice"] = frame["SalePrice"].astype(float)
    frame.loc[0, "SalePrice"] = value
    with pytest.raises(AmesSchemaError, match=message):
        validate_ames(frame)


def test_target_is_optional_for_prediction_inputs(ames_frame: pd.DataFrame) -> None:
    frame = ames_frame.drop(columns=["SalePrice"])
    validate_ames(frame, require_target=False)
    with pytest.raises(AmesSchemaError, match="SalePrice"):
        validate_ames(frame)


def test_duplicate_columns_are_rejected(ames_frame: pd.DataFrame) -> None:
    frame = pd.concat([ames_frame, ames_frame[["LotArea"]]], axis=1)
    with pytest.raises(AmesSchemaError, match="Duplicate column name.*LotArea"):
        validate_ames(frame)


def test_all_problems_are_reported_together(ames_frame: pd.DataFrame) -> None:
    frame = ames_frame.drop(columns=["Street"])
    frame["LotArea"] = "unknown"
    with pytest.raises(AmesSchemaError) as raised:
        validate_ames(frame)
    assert "Street" in str(raised.value) and "LotArea" in str(raised.value)


def test_load_from_explicit_path(ames_csv: Path, ames_frame: pd.DataFrame) -> None:
    frame = load_ames(ames_csv)
    assert frame.shape == ames_frame.shape


def test_explicit_missing_path_is_never_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_module.requests, "get", pytest.fail)
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_ames(tmp_path / "missing.csv")


def test_load_without_validation(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert load_ames(path, validate=False).shape == (1, 2)
    with pytest.raises(AmesSchemaError):
        load_ames(path)


def test_default_path_is_downloaded_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ames_csv: Path
) -> None:
    downloaded: list[Path] = []

    def fake_download(destination: Path) -> Path:
        downloaded.append(destination)
        destination.parent.mkdir(parents=True)
        destination.write_bytes(ames_csv.read_bytes())
        return destination

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_module, "download_ames", fake_download)
    frame = load_ames()
    assert downloaded == [Path("data") / "train.csv"]
    assert len(frame) == 200


def test_default_path_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="download=False"):
        load_ames(download=False)


def test_download_verifies_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"Id,SalePrice\n1,100\n"
    monkeypatch.setattr(data_module.requests, "get", fake_get(content))
    target = download_ames(
        tmp_path / "sub" / "train.csv", sha256=hashlib.sha256(content).hexdigest()
    )
    assert target.read_bytes() == content
    assert not (tmp_path / "sub" / "train.csv.part").exists()


def test_checksum_mismatch_saves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module.requests, "get", fake_get(b"tampered"))
    target = tmp_path / "train.csv"
    with pytest.raises(DatasetDownloadError, match="Checksum mismatch"):
        download_ames(target)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_network_errors_are_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(url: str, timeout: float) -> FakeResponse:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(data_module.requests, "get", broken)
    with pytest.raises(DatasetDownloadError, match="offline"):
        download_ames(tmp_path / "train.csv")


def test_http_errors_are_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module.requests, "get", fake_get(b"", status=404))
    with pytest.raises(DatasetDownloadError, match="404"):
        download_ames(tmp_path / "train.csv")


def test_existing_file_is_kept_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "train.csv"
    target.write_bytes(b"original")
    monkeypatch.setattr(data_module.requests, "get", fake_get(b"new"))
    assert download_ames(target).read_bytes() == b"original"
    assert download_ames(target, sha256=None, force=True).read_bytes() == b"new"


def test_pinned_checksum_matches_the_repository_copy() -> None:
    real_data = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    if not real_data.exists():
        pytest.skip("data/train.csv is not available")
    assert hashlib.sha256(real_data.read_bytes()).hexdigest() == DATASET_SHA256


def test_split_sizes_and_disjointness(ames_frame: pd.DataFrame) -> None:
    train, test = train_test_split_df(ames_frame, test_size=0.25, random_state=1)
    assert (len(train), len(test)) == (150, 50)
    assert set(train.index).isdisjoint(test.index)
    again, _ = train_test_split_df(ames_frame, test_size=0.25, random_state=1)
    assert train.index.equals(again.index)


def test_stratified_split_keeps_class_balance(ames_frame: pd.DataFrame) -> None:
    frame = ames_frame.assign(label=(ames_frame["SalePrice"] >= 250_000).astype(int))
    _, test = train_test_split_df(frame, test_size=0.3, random_state=0, stratify="label")
    overall, held_out = frame["label"].mean(), test["label"].mean()
    assert abs(overall - held_out) < 0.02


def test_fingerprint_is_stable_and_content_sensitive(ames_frame: pd.DataFrame) -> None:
    tiny = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    expected = hashlib.sha256(b"a,b\n1,x\n2,\n").hexdigest()
    assert dataset_fingerprint(tiny) == expected
    assert dataset_fingerprint(tiny.set_axis([10, 11])) == expected  # index is ignored
    changed = ames_frame.copy()
    changed.loc[0, "LotArea"] += 1
    assert dataset_fingerprint(changed) != dataset_fingerprint(ames_frame)
