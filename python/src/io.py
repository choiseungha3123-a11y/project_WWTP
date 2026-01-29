"""
데이터 로드 및 전처리 모듈
CSV 파일 로드, 시간 인덱스 설정, 데이터 병합
"""

from pathlib import Path
import pandas as pd


def load_csvs(data_root: str):
    """루트 디렉토리에서 FLOW/TMS/AWS CSV 파일들을 로드합니다."""
    root = Path(data_root)
    df_flow = pd.read_csv(root / "FLOW_Actual.csv")
    df_tms = pd.read_csv(root / "TMS_Actual.csv")
    df_aws_368 = pd.read_csv(root / "AWS_368.csv")
    df_aws_541 = pd.read_csv(root / "AWS_541.csv")
    df_aws_569 = pd.read_csv(root / "AWS_569.csv")
    return df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569


def set_datetime_index(df, time_col, tz=None):
    """시간 컬럼을 DatetimeIndex로 설정"""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).set_index(time_col).sort_index()
    return out


def summarize_available_period(dfs):
    """각 데이터소스의 사용 가능한 기간 요약"""
    rows = []
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            rows.append([name, None, None, 0])
        else:
            rows.append([name, df.index.min(), df.index.max(), len(df)])
    return pd.DataFrame(rows, columns=["source", "start", "end", "n_rows"])


def merge_sources_on_time(dfs, how="outer"):
    """시간 인덱스 기준으로 여러 데이터소스 병합"""
    items = [df.copy() for df in dfs.values() if df is not None and len(df) > 0]
    if not items:
        raise ValueError("병합할 비어있지 않은 데이터프레임이 없습니다.")
    out = items[0]
    for nxt in items[1:]:
        out = out.join(nxt, how=how)
    out = out.sort_index()
    return out


def prep_flow(df_flow: pd.DataFrame) -> pd.DataFrame:
    """FLOW 데이터 전처리: TankA + TankB = Q_in"""
    out = df_flow.copy()
    for c in ["flow_TankA", "flow_TankB"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "flow_TankA" in out.columns and "flow_TankB" in out.columns:
        out["Q_in"] = out["flow_TankA"] + out["flow_TankB"]
        # 레벨 컬럼은 유지; 원본 flow 컬럼은 타겟 유출 방지를 위해 제거
        out = out.drop(columns=["flow_TankA", "flow_TankB"])
    return out


def prep_aws_station(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """AWS 관측소 데이터 전처리 및 suffix 추가"""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    return out.add_suffix(suffix)


def prep_aws(df_aws_368, df_aws_541, df_aws_569) -> pd.DataFrame:
    """3개 AWS 관측소 데이터 병합"""
    aws368 = prep_aws_station(df_aws_368, "_368")
    aws541 = prep_aws_station(df_aws_541, "_541")
    aws569 = prep_aws_station(df_aws_569, "_569")
    df_aws = aws368.merge(aws541, left_index=True, right_index=True, how="inner").merge(
        aws569, left_index=True, right_index=True, how="inner"
    )
    df_aws["datetime"] = df_aws.index
    return df_aws
