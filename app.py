from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from bayser_app.runner import build_bayser_command, stream_bayser_run


st.set_page_config(
    page_title="Bayser",
    layout="wide",
)

st.title("Bayser")
st.caption("Online frontend for radiocarbon-informed Bayesian seriation")


APP_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = APP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

ONLINE_TIMEOUT_SECONDS = 120
OUTPUT_RETENTION_SECONDS = 2 * 60 * 60

DEFAULT_INTCAL20_CANDIDATES = [
    APP_ROOT / "bayser_app" / "assets" / "intcal20.14c",
    APP_ROOT / "assets" / "intcal20.14c",
    APP_ROOT / "data" / "intcal20.14c",
    APP_ROOT / "intcal20.14c",
]

DEFAULT_INTCAL20_PATH = next(
    (p for p in DEFAULT_INTCAL20_CANDIDATES if p.exists()),
    None,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def tidy_value(x: Any) -> str:
    try:
        val = float(x)
        if val.is_integer():
            return str(int(val))
        return f"{val:.3f}"
    except (TypeError, ValueError):
        return str(x)


def read_csv_preview(uploaded_file, *, index_col=None) -> pd.DataFrame:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, index_col=index_col)
    uploaded_file.seek(0)
    return df


def read_uploaded_csv_raw(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        return df
    except Exception:
        return None


def read_progress(progress_path: Path) -> dict | None:
    """Read Bayser's sampler progress file.

    The file may be read while Bayser is writing it. In that case JSON parsing
    can fail transiently; returning None is safer than surfacing a UI error.
    """

    if not progress_path.exists():
        return None

    try:
        text = progress_path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except Exception:
        return None


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "–"

    seconds = int(round(seconds))

    if seconds < 60:
        return f"{seconds}s"

    minutes, sec = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes}m {sec:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def progress_info(payload: dict | None, elapsed: float) -> tuple[int, str]:
    """Convert Bayser progress metadata into one compact progress line."""

    if payload is None:
        return 0, f"Preparing model · elapsed {format_seconds(elapsed)} · remaining –"

    status = str(payload.get("status", "running"))
    phase = str(payload.get("phase") or "sampling")

    done = payload.get("done")
    total = payload.get("total")

    percent = float(payload.get("percent", 0.0))
    value = max(0, min(100, int(round(percent))))

    remaining_text = "–"

    if done is not None and total is not None:
        done_f = float(done)
        total_f = float(total)

        if done_f > 0 and elapsed > 0 and total_f >= done_f:
            rate = done_f / elapsed
            remaining = (total_f - done_f) / rate if rate > 0 else None
            remaining_text = format_seconds(remaining)

        if status == "finished":
            return 100, (
                f"Sampling finished · {int(total_f)}/{int(total_f)} sampler steps "
                f"· elapsed {format_seconds(elapsed)}"
            )

        if status == "failed":
            return value, (
                f"Sampling failed · {int(done_f)}/{int(total_f)} sampler steps "
                f"({value}%) · elapsed {format_seconds(elapsed)}"
            )

        return value, (
            f"{phase.capitalize()} · {int(done_f)}/{int(total_f)} sampler steps "
            f"({value}%) · elapsed {format_seconds(elapsed)} "
            f"· remaining {remaining_text}"
        )

    return value, (
        f"{phase.capitalize()} · {value}% "
        f"· elapsed {format_seconds(elapsed)} · remaining {remaining_text}"
    )


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cleanup_old_outputs(
    output_root: Path,
    *,
    max_age_seconds: int = OUTPUT_RETENTION_SECONDS,
    keep: set[Path] | None = None,
) -> int:
    """Remove old run directories and zip archives from the output folder."""

    if not output_root.exists():
        return 0

    keep_resolved = {p.resolve() for p in (keep or set())}
    now = time.time()
    removed = 0

    for path in output_root.iterdir():
        if path.resolve() in keep_resolved:
            continue

        if path.is_dir() and not path.name.startswith("run_"):
            continue

        if path.is_file() and path.suffix != ".zip":
            continue

        try:
            age = now - path.stat().st_mtime
        except FileNotFoundError:
            continue

        if age < max_age_seconds:
            continue

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except OSError:
            # The file may still be in use. It can be removed during a later run.
            continue

    return removed


def convert_mostly_numeric_columns(
    df: pd.DataFrame,
    *,
    id_columns: set[str] | None = None,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Convert mostly numeric columns while preserving text columns."""

    id_columns = id_columns or set()
    out = df.copy()

    for col in out.columns:
        if col in id_columns:
            continue

        converted = pd.to_numeric(out[col], errors="coerce")
        original_non_missing = out[col].notna().sum()
        converted_non_missing = converted.notna().sum()

        if (
            original_non_missing > 0
            and converted_non_missing / original_non_missing >= threshold
        ):
            out[col] = converted

    return out


def round_numeric_columns(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def dataframe_if_exists(path: Path, title: str) -> None:
    if path.exists():
        st.subheader(title)
        st.dataframe(pd.read_csv(path), width="stretch")


def parse_outlier_id(spec: str) -> str:
    """Extract the assemblage ID from ASSEMBLAGE_ID, ASSEMBLAGE_ID:PRIOR, etc."""

    spec = str(spec).strip()
    for sep in [":", "=", ","]:
        if sep in spec:
            return spec.split(sep, 1)[0].strip()
    return spec


def get_feature_ids(
    features_raw: pd.DataFrame | None,
    feature_id_col: str,
) -> list[str]:
    if features_raw is None:
        return []

    feature_id_col = feature_id_col.strip()

    if feature_id_col:
        if feature_id_col not in features_raw.columns:
            return []
        return features_raw[feature_id_col].astype(str).tolist()

    # Bayser will use the first column as row index if no explicit ID column is
    # supplied. Mirror this behaviour for validation.
    if len(features_raw.columns) == 0:
        return []

    return features_raw.iloc[:, 0].astype(str).tolist()


def validate_inputs(
    *,
    features_file,
    feature_id_col: str,
    c14_file,
    c14_id_col: str,
    bp_col: str,
    error_col: str,
    outlier_specs: list[str],
    use_c14: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Return errors, warnings, and retained feature IDs for pre-run checks."""

    errors: list[str] = []
    warnings: list[str] = []

    features_raw = read_uploaded_csv_raw(features_file)
    c14_raw = read_uploaded_csv_raw(c14_file)

    if features_file is None:
        errors.append("Please upload a feature matrix CSV.")
        return errors, warnings, []

    if features_raw is None:
        errors.append("The feature matrix CSV could not be read.")
        return errors, warnings, []

    feature_id_col = feature_id_col.strip()

    if feature_id_col and feature_id_col not in features_raw.columns:
        errors.append(
            f"Feature ID column `{feature_id_col}` does not exist in the feature matrix."
        )

    feature_ids = get_feature_ids(features_raw, feature_id_col)
    feature_id_set = set(feature_ids)

    if not feature_ids:
        errors.append("Could not determine assemblage IDs from the feature matrix.")

    if len(feature_ids) != len(feature_id_set):
        warnings.append(
            "The feature matrix appears to contain duplicate assemblage IDs."
        )

    if use_c14:
        if c14_file is None:
            errors.append("Radiocarbon-linked mode requires a radiocarbon table.")

        if c14_raw is None:
            errors.append("The radiocarbon table could not be read.")
        else:
            required_c14_cols = {
                "14C ID column": c14_id_col.strip(),
                "Radiocarbon age column": bp_col.strip(),
                "Radiocarbon error column": error_col.strip(),
            }

            for label, col in required_c14_cols.items():
                if not col:
                    errors.append(f"{label} is required for radiocarbon-linked mode.")
                elif col not in c14_raw.columns:
                    errors.append(
                        f"{label} `{col}` does not exist in the radiocarbon table."
                    )

            if c14_id_col.strip() in c14_raw.columns and feature_id_set:
                c14_ids = set(c14_raw[c14_id_col.strip()].astype(str))
                unmatched = sorted(c14_ids - feature_id_set)

                if unmatched:
                    preview = ", ".join(unmatched[:8])
                    more = "..." if len(unmatched) > 8 else ""
                    warnings.append(
                        f"{len(unmatched)} radiocarbon ID(s) are not present in the "
                        f"feature matrix: {preview}{more}"
                    )

    if outlier_specs:
        outlier_ids = [parse_outlier_id(spec) for spec in outlier_specs]
        missing_outliers = sorted(set(outlier_ids) - feature_id_set)

        if missing_outliers:
            preview = ", ".join(missing_outliers[:8])
            more = "..." if len(missing_outliers) > 8 else ""
            errors.append(
                f"Outlier candidate ID(s) not found in the feature matrix: "
                f"{preview}{more}"
            )

        if use_c14 and c14_raw is not None and c14_id_col.strip() in c14_raw.columns:
            c14_ids = set(c14_raw[c14_id_col.strip()].astype(str))
            outliers_without_c14 = sorted(set(outlier_ids) - c14_ids)

            if outliers_without_c14:
                preview = ", ".join(outliers_without_c14[:8])
                more = "..." if len(outliers_without_c14) > 8 else ""
                errors.append(
                    f"Outlier candidate ID(s) have no matching radiocarbon "
                    f"determination: {preview}{more}"
                )

    return errors, warnings, feature_ids


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------


with st.sidebar:
    st.header("Sampling preset")

    preset = st.selectbox(
        "Preset",
        ["quick"],
        index=0,
        help="The public online version is limited to the quick preset.",
    )

    st.caption(
        "In the online version, only the limited quick preset is available. "
        "Standard and careful runs should be executed locally."
    )

    st.markdown(
        """
        <div style="color: #999; font-size: 0.9em; line-height: 1.4;">
        standard — disabled online<br>
        careful — disabled online
        </div>
        """,
        unsafe_allow_html=True,
    )

    draws, tune, chains, target_accept, max_treedepth = 500, 500, 2, 0.95, 12

    st.header("Filtering")

    filter_data = st.checkbox("Apply filtering", value=True)
    min_type_count = st.number_input("Minimum type count", min_value=1, value=2)
    min_grave_count = st.number_input("Minimum assemblage count", min_value=1, value=2)

    st.header("Model")

    include_richness = st.checkbox("Include richness effect", value=True)
    repulsion_strength = st.number_input(
        "Repulsion strength",
        min_value=0.0,
        value=0.25,
        step=0.05,
    )
    random_seed = st.number_input("Random seed", value=123, step=1)

    st.header("Runtime")

    st.caption(
        "Online runs are limited to 2 minutes. For longer runs, use Bayser locally."
    )

    show_live_log = st.checkbox("Show live log", value=False)


tab_data, tab_run, tab_results = st.tabs(["Data", "Run", "Results"])


# -----------------------------------------------------------------------------
# Data tab
# -----------------------------------------------------------------------------


with tab_data:
    st.header("Input data")

    st.info(
        """
        **Standard workflow**

        1. Upload a binary assemblage-by-type feature matrix.
        2. Optionally upload a radiocarbon table. If a radiocarbon table is supplied,
           Bayser links the inferred seriation axis to calendar time using IntCal20.
        3. Check the column mapping and data preview.
        4. Run the limited online model and inspect posterior ranks, diagnostic plots,
           and possible typology–radiocarbon tension.
        5. Download the complete run for local inspection.
        """
    )

    features_file = st.file_uploader(
        "Feature matrix CSV",
        type=["csv"],
        help="Rows should be assemblages, columns should be artefact types.",
    )

    feature_id_col = st.text_input(
        "Feature ID column",
        value="",
        help="Leave empty if the first column should be used as row index.",
    )

    c14_file = st.file_uploader(
        "Optional radiocarbon table CSV",
        type=["csv"],
    )

    st.subheader("Calibration curve")

    if DEFAULT_INTCAL20_PATH is not None:
        st.success("Default IntCal20 curve available (IntCal20; Reimer et al. 2020).")
    else:
        st.warning(
            "No bundled IntCal20 curve was found. Upload an IntCal20 file for "
            "radiocarbon-linked runs."
        )

    intcal20_file = st.file_uploader(
        "Optional custom IntCal20 calibration curve",
        type=["14c", "csv", "txt"],
        help="Only needed if you want to override the bundled IntCal20 curve.",
    )

    c14_id_col = ""
    bp_col = ""
    error_col = ""

    if c14_file is not None:
        st.subheader("Radiocarbon column mapping")

        c14_id_col = st.text_input("14C ID column", value="grave_id")
        bp_col = st.text_input("Radiocarbon age column", value="bp")
        error_col = st.text_input("Radiocarbon error column", value="error")

    if features_file is not None:
        try:
            if feature_id_col.strip():
                features_preview = read_csv_preview(features_file)
            else:
                features_preview = read_csv_preview(features_file, index_col=0)

            st.subheader("Feature matrix preview")
            st.write(
                f"{features_preview.shape[0]} assemblages × "
                f"{features_preview.shape[1]} artefact types"
            )
            st.dataframe(features_preview.head(20), width="stretch")

            features_raw_for_check = read_uploaded_csv_raw(features_file)

            if feature_id_col.strip():
                if (
                    features_raw_for_check is not None
                    and feature_id_col.strip() in features_raw_for_check.columns
                ):
                    st.success(f"Feature ID column `{feature_id_col.strip()}` found.")
                else:
                    st.error(
                        f"Feature ID column `{feature_id_col.strip()}` was not found."
                    )
            else:
                st.caption(
                    "No feature ID column supplied; the first column will be used "
                    "as assemblage ID."
                )

        except Exception as e:
            st.error(f"Could not read feature matrix: {e}")

    if c14_file is not None:
        try:
            c14_preview = read_csv_preview(c14_file)
            st.subheader("Radiocarbon table preview")
            st.write(f"{c14_preview.shape[0]} rows × {c14_preview.shape[1]} columns")
            st.dataframe(c14_preview.head(20), width="stretch")

            mapped_cols = [
                ("14C ID column", c14_id_col),
                ("Radiocarbon age column", bp_col),
                ("Radiocarbon error column", error_col),
            ]

            missing = [
                f"{label}: `{col}`"
                for label, col in mapped_cols
                if col.strip() and col.strip() not in c14_preview.columns
            ]

            if missing:
                st.error("Missing mapped column(s): " + "; ".join(missing))
            else:
                st.success("Radiocarbon column mapping looks valid.")

        except Exception as e:
            st.error(f"Could not read radiocarbon table: {e}")


# -----------------------------------------------------------------------------
# Run tab
# -----------------------------------------------------------------------------


with tab_run:
    st.header("Run Bayser")

    bundled_or_uploaded_curve_available = (
        intcal20_file is not None or DEFAULT_INTCAL20_PATH is not None
    )
    use_c14 = c14_file is not None and bundled_or_uploaded_curve_available

    st.write("Mode:", "**radiocarbon-linked**" if use_c14 else "**typology-only**")
    st.write("Preset:", preset)
    st.write("Draws per chain:", draws)
    st.write("Tuning steps per chain:", tune)
    st.write("Chains:", chains)
    st.write("Timeout:", f"{ONLINE_TIMEOUT_SECONDS} seconds")

    if c14_file is not None and not bundled_or_uploaded_curve_available:
        st.error(
            "A radiocarbon table was uploaded, but no IntCal20 calibration curve "
            "is available. Upload a calibration curve or add a bundled IntCal20 file."
        )

    outlier_specs: list[str] = []
    outlier_all: float | None = None

    if use_c14:
        st.subheader("Optional outlier candidates")

        outlier_text = st.text_area(
            "One outlier specification per line",
            value="",
            placeholder="ASSEMBLAGE_ID:0.5",
            help="Use ASSEMBLAGE_ID or ASSEMBLAGE_ID:PRIOR.",
        )

        outlier_specs = [
            line.strip()
            for line in outlier_text.splitlines()
            if line.strip()
        ]

        use_outlier_all = st.checkbox(
            "Assign low outlier prior to all dated assemblages",
            value=False,
        )

        if use_outlier_all:
            outlier_all = st.number_input(
                "Global outlier prior",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
            )

    validation_errors, validation_warnings, feature_ids = validate_inputs(
        features_file=features_file,
        feature_id_col=feature_id_col,
        c14_file=c14_file,
        c14_id_col=c14_id_col,
        bp_col=bp_col,
        error_col=error_col,
        outlier_specs=outlier_specs,
        use_c14=use_c14,
    )

    if validation_warnings:
        with st.expander("Input warnings", expanded=False):
            for warning in validation_warnings:
                st.warning(warning)

    if validation_errors:
        with st.expander("Input checks", expanded=True):
            for error in validation_errors:
                st.error(error)
    elif features_file is not None:
        st.success("Input checks passed.")

    run_clicked = st.button(
        "Run Bayser",
        type="primary",
        disabled=features_file is None
        or bool(validation_errors)
        or (c14_file is not None and not bundled_or_uploaded_curve_available),
    )

    if run_clicked:
        keep_outputs: set[Path] = set()
        if st.session_state.get("last_run_root"):
            keep_outputs.add(Path(st.session_state["last_run_root"]))

        cleanup_old_outputs(
            OUTPUT_ROOT,
            max_age_seconds=OUTPUT_RETENTION_SECONDS,
            keep=keep_outputs,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = OUTPUT_ROOT / f"run_{timestamp}"
        input_dir = run_root / "input"
        results_dir = run_root / "results"
        plot_dir = run_root / "plots"

        input_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        progress_path = results_dir / "progress.json"

        features_path = input_dir / "features.csv"
        features_path.write_bytes(features_file.getvalue())

        c14_path = None
        intcal20_path = None

        if c14_file is not None:
            c14_path = input_dir / "c14.csv"
            c14_path.write_bytes(c14_file.getvalue())

        if intcal20_file is not None:
            intcal20_path = input_dir / "intcal20.14c"
            intcal20_path.write_bytes(intcal20_file.getvalue())
        elif use_c14 and DEFAULT_INTCAL20_PATH is not None:
            intcal20_path = input_dir / "intcal20.14c"
            shutil.copyfile(DEFAULT_INTCAL20_PATH, intcal20_path)

        cmd = build_bayser_command(
            features_path=features_path,
            results_dir=results_dir,
            plot_dir=plot_dir,
            c14_path=c14_path,
            intcal20_path=intcal20_path,
            feature_id_col=feature_id_col.strip() or None,
            c14_id_col=c14_id_col.strip() or None,
            bp_col=bp_col.strip() or None,
            error_col=error_col.strip() or None,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=int(random_seed),
            max_treedepth=max_treedepth,
            min_type_count=int(min_type_count),
            min_grave_count=int(min_grave_count),
            include_richness=include_richness,
            filter_data=filter_data,
            repulsion_strength=float(repulsion_strength),
            outliers=outlier_specs,
            outlier_all=outlier_all,
        )

        (run_root / "command.txt").write_text(
            " ".join(cmd),
            encoding="utf-8",
        )

        app_config = {
            "preset": preset,
            "use_c14": use_c14,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "max_treedepth": max_treedepth,
            "filter": filter_data,
            "min_type_count": min_type_count,
            "min_grave_count": min_grave_count,
            "include_richness": include_richness,
            "repulsion_strength": repulsion_strength,
            "random_seed": random_seed,
            "outliers": outlier_specs,
            "outlier_all": outlier_all,
            "timeout_seconds": ONLINE_TIMEOUT_SECONDS,
            "progress_file": str(progress_path),
            "default_intcal20": str(DEFAULT_INTCAL20_PATH)
            if DEFAULT_INTCAL20_PATH is not None
            else None,
            "custom_intcal20_uploaded": intcal20_file is not None,
        }

        write_json(run_root / "app_config.json", app_config)

        with st.expander("Command"):
            st.code(" ".join(cmd), language="bash")

        st.session_state["last_run_root"] = str(run_root)

        progress = st.progress(0, text="Starting Bayser...")
        log_box = st.empty()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        final_event: dict | None = None

        env = {
            "BAYSER_PROGRESS_FILE": str(progress_path),
        }

        for event in stream_bayser_run(
            cmd,
            timeout_seconds=ONLINE_TIMEOUT_SECONDS,
            env=env,
        ):
            elapsed = float(event.get("elapsed", 0.0))

            progress_payload = read_progress(progress_path)
            value, text = progress_info(progress_payload, elapsed)
            progress.progress(value, text=text)

            if event["type"] == "line":
                stream_name = event["stream"]
                line = event["text"]

                if stream_name == "stdout":
                    stdout_lines.append(line)
                else:
                    stderr_lines.append(line)

                if show_live_log:
                    combined_tail = "".join((stdout_lines + stderr_lines)[-40:])
                    log_box.code(combined_tail or "Waiting for output...")

            elif event["type"] in {"timeout", "done"}:
                final_event = event
                break

            time.sleep(0.05)

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)

        if final_event is not None:
            stdout_text = final_event.get("stdout", stdout_text)
            stderr_text = final_event.get("stderr", stderr_text)

        (run_root / "stdout.txt").write_text(stdout_text, encoding="utf-8")
        (run_root / "stderr.txt").write_text(stderr_text, encoding="utf-8")

        final_progress = read_progress(progress_path)

        if final_progress is not None:
            write_json(run_root / "final_progress.json", final_progress)

        if final_event is None:
            progress.empty()
            st.error("Bayser stopped without returning a final status.")

        elif final_event["type"] == "timeout":
            progress.empty()
            st.error(f"Bayser timed out after {format_seconds(final_event['elapsed'])}.")

            with st.expander("stderr", expanded=True):
                st.code(stderr_text or "(empty)")

            with st.expander("stdout"):
                st.code(stdout_text or "(empty)")

        elif final_event["returncode"] != 0:
            progress.empty()
            st.error("Bayser failed.")

            with st.expander("stderr", expanded=True):
                st.code(stderr_text or "(empty)")

            with st.expander("stdout"):
                st.code(stdout_text or "(empty)")

        else:
            progress.progress(
                100,
                text=f"Bayser completed · elapsed {format_seconds(final_event['elapsed'])}",
            )

            st.success(
                f"Bayser completed in {format_seconds(final_event['elapsed'])}. "
                "Open the Results tab."
            )

            if show_live_log:
                with st.expander("Run log"):
                    combined = (stdout_text + "\n" + stderr_text).strip()
                    st.code(combined or "(no output)")


# -----------------------------------------------------------------------------
# Results tab
# -----------------------------------------------------------------------------


with tab_results:
    st.header("Results")

    run_root_value = st.session_state.get("last_run_root")

    if not run_root_value:
        st.info("No run completed in this session yet.")
    else:
        run_root = Path(run_root_value)
        results_dir = run_root / "results"
        plot_dir = run_root / "plots"

        st.caption(f"Run directory: `{run_root}`")

        # ---------------------------------------------------------------------
        # Compact run summary
        # ---------------------------------------------------------------------

        metadata_path = results_dir / "metadata.csv"
        meta = None
        meta_dict = {}

        if metadata_path.exists():
            meta = pd.read_csv(metadata_path)
            meta_dict = dict(zip(meta["setting"].astype(str), meta["value"]))

        if meta_dict:
            st.subheader("Run summary")

            summary_df = pd.DataFrame(
                [
                    {
                        "Assemblages": tidy_value(meta_dict.get("n_graves", "–")),
                        "Types": tidy_value(meta_dict.get("n_types", "–")),
                        "C14 dates": tidy_value(meta_dict.get("n_c14_finite", "–")),
                        "Mode": str(meta_dict.get("chronology_mode", "–")),
                        "Divergences": tidy_value(meta_dict.get("divergences", "–")),
                        "Spearman vs CA/RA": tidy_value(
                            meta_dict.get("pymc_ra_spearman_abs", "–")
                        ),
                        "Pearson vs CA/RA": tidy_value(
                            meta_dict.get("pymc_ra_pearson_abs", "–")
                        ),
                    }
                ]
            )

            st.table(summary_df)

        # ---------------------------------------------------------------------
        # Main plot
        # ---------------------------------------------------------------------

        st.subheader("Main diagnostic plot")

        preferred_plots = [
            plot_dir / "dataset_posterior_rank_distributions.png",
            plot_dir / "dataset_calendar_ages_along_pymc_order.png",
            plot_dir / "dataset_pymc_posterior_order.png",
        ]

        main_plot = next((p for p in preferred_plots if p.exists()), None)

        if main_plot is not None:
            st.image(str(main_plot), caption=main_plot.name)
        else:
            st.info("No main plot found.")

        # ---------------------------------------------------------------------
        # Essential outlier information, if available
        # ---------------------------------------------------------------------

        posthoc_path = results_dir / "posthoc_outlier_candidates.csv"
        active_outliers_path = results_dir / "active_outliers.csv"

        if posthoc_path.exists():
            posthoc = pd.read_csv(posthoc_path)

            if not posthoc.empty:
                st.subheader("Post-hoc outlier candidates")

                preferred_cols = [
                    "grave_id",
                    "posterior_rank",
                    "posthoc_outlier_suggestion",
                    "suggested_cli_arg",
                    "unmodelled_cal_bp_mean",
                    "expected_cal_bp_mean",
                    "posterior_cal_bp_mean",
                    "shift_unmodelled_vs_expected",
                    "shift_model_vs_expected",
                ]

                show_cols = [c for c in preferred_cols if c in posthoc.columns]
                posthoc_show = posthoc[show_cols].head(8).copy()

                posthoc_show = posthoc_show.rename(
                    columns={
                        "grave_id": "Assemblage",
                        "posterior_rank": "Rank",
                        "posthoc_outlier_suggestion": "Suggestion",
                        "suggested_cli_arg": "Suggested rerun",
                        "unmodelled_cal_bp_mean": "Single-date cal BP",
                        "expected_cal_bp_mean": "Typological cal BP",
                        "posterior_cal_bp_mean": "Modelled cal BP",
                        "shift_unmodelled_vs_expected": "Single-date − typological",
                        "shift_model_vs_expected": "Modelled − typological",
                    }
                )

                posthoc_show = convert_mostly_numeric_columns(
                    posthoc_show,
                    id_columns={"Assemblage", "Suggestion", "Suggested rerun"},
                )
                posthoc_show = round_numeric_columns(posthoc_show, digits=1)

                st.dataframe(
                    posthoc_show,
                    width="stretch",
                    hide_index=True,
                )

        if active_outliers_path.exists():
            active = pd.read_csv(active_outliers_path)

            if not active.empty:
                st.subheader("Active outlier model")

                preferred_cols = [
                    "grave_id",
                    "posterior_rank",
                    "p_outlier_mean",
                    "p_outlier_hdi_3",
                    "p_outlier_hdi_97",
                    "expected_cal_bp_mean",
                    "posterior_cal_bp_mean",
                    "unmodelled_cal_bp_mean",
                    "shift_from_typological_expectation",
                    "shift_from_unmodelled_calibration",
                ]

                show_cols = [c for c in preferred_cols if c in active.columns]
                active_show = active[show_cols].copy()

                if "p_outlier_mean" in active_show.columns:
                    active_show = active_show.sort_values(
                        "p_outlier_mean",
                        ascending=False,
                    )

                active_show = active_show.head(8)

                active_show = active_show.rename(
                    columns={
                        "grave_id": "Assemblage",
                        "posterior_rank": "Rank",
                        "p_outlier_mean": "Mean p(outlier)",
                        "p_outlier_hdi_3": "p(outlier) HDI 3%",
                        "p_outlier_hdi_97": "p(outlier) HDI 97%",
                        "expected_cal_bp_mean": "Typological cal BP",
                        "posterior_cal_bp_mean": "Modelled cal BP",
                        "unmodelled_cal_bp_mean": "Single-date cal BP",
                        "shift_from_typological_expectation": "Modelled − typological",
                        "shift_from_unmodelled_calibration": "Modelled − single-date",
                    }
                )

                active_show = convert_mostly_numeric_columns(
                    active_show,
                    id_columns={"Assemblage"},
                )

                probability_cols = [
                    "Mean p(outlier)",
                    "p(outlier) HDI 3%",
                    "p(outlier) HDI 97%",
                ]

                calendar_cols = [
                    "Typological cal BP",
                    "Modelled cal BP",
                    "Single-date cal BP",
                    "Modelled − typological",
                    "Modelled − single-date",
                ]

                for col in probability_cols:
                    if col in active_show.columns:
                        active_show[col] = active_show[col].round(3)

                for col in calendar_cols:
                    if col in active_show.columns:
                        active_show[col] = active_show[col].round(1)

                if "Rank" in active_show.columns:
                    active_show["Rank"] = active_show["Rank"].round(1)

                st.dataframe(
                    active_show,
                    width="stretch",
                    hide_index=True,
                )

        # ---------------------------------------------------------------------
        # Essential table: assemblage summary
        # ---------------------------------------------------------------------

        grave_summary_path = results_dir / "grave_summary.csv"

        if grave_summary_path.exists():
            st.subheader("Assemblage summary")

            grave = pd.read_csv(grave_summary_path)

            preferred_cols = [
                "grave_id",
                "posterior_rank",
                "posterior_rank_mean",
                "posterior_rank_sd",
                "posterior_rank_hdi_3",
                "posterior_rank_hdi_97",
                "ra_rank",
                "rank_difference_pymc_minus_ra",
                "unmodelled_cal_bp_mean",
                "expected_cal_bp_mean",
                "posterior_cal_bp_mean",
                "p_outlier_mean",
            ]

            show_cols = [c for c in preferred_cols if c in grave.columns]

            if show_cols:
                grave_show = grave[show_cols].copy()
            else:
                grave_show = grave.copy()

            grave_show = convert_mostly_numeric_columns(
                grave_show,
                id_columns={"grave_id"},
            )
            grave_show = round_numeric_columns(grave_show, digits=3)

            st.dataframe(grave_show, width="stretch")
        else:
            st.info("No assemblage summary found.")

        # ---------------------------------------------------------------------
        # Download
        # ---------------------------------------------------------------------

        st.subheader("Download")

        zip_path = shutil.make_archive(str(run_root), "zip", run_root)

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download complete run",
                data=f,
                file_name=f"{run_root.name}.zip",
                mime="application/zip",
                type="primary",
            )

        # ---------------------------------------------------------------------
        # Secondary outputs
        # ---------------------------------------------------------------------

        with st.expander("Additional plots"):
            plot_files = sorted(plot_dir.glob("*.png"))

            if not plot_files:
                st.info("No PNG plots found.")
            else:
                for plot_file in plot_files:
                    if main_plot is not None and plot_file == main_plot:
                        continue
                    st.image(str(plot_file), caption=plot_file.name)

        with st.expander("Additional tables"):
            secondary_tables = [
                ("Type summary", results_dir / "type_summary.csv"),
                ("Score comparison", results_dir / "score_comparison.csv"),
                ("Chain diagnostics", results_dir / "chain_diagnostics.csv"),
                ("Parameter diagnostics", results_dir / "parameter_diagnostics.csv"),
                ("Unmodelled calibration", results_dir / "unmodelled_calibration.csv"),
            ]

            found_any = False

            for title, path in secondary_tables:
                if path.exists():
                    found_any = True
                    st.subheader(title)
                    df = pd.read_csv(path)
                    st.dataframe(df, width="stretch")

            if not found_any:
                st.info("No additional tables found.")

        with st.expander("Run metadata and logs"):
            command_path = run_root / "command.txt"
            if command_path.exists():
                st.subheader("Command")
                st.code(command_path.read_text(encoding="utf-8"), language="bash")

            if meta is not None:
                st.subheader("Metadata")
                meta_show = meta.copy()
                meta_show["value"] = meta_show["value"].map(tidy_value)
                st.dataframe(meta_show, width="stretch")

            progress_path = results_dir / "progress.json"
            final_progress_path = run_root / "final_progress.json"

            if final_progress_path.exists():
                st.subheader("Final sampler progress metadata")
                st.json(json.loads(final_progress_path.read_text(encoding="utf-8")))
            elif progress_path.exists():
                st.subheader("Sampler progress metadata")
                st.json(json.loads(progress_path.read_text(encoding="utf-8")))

            stdout_path = run_root / "stdout.txt"
            stderr_path = run_root / "stderr.txt"

            if stderr_path.exists():
                st.subheader("stderr")
                st.code(stderr_path.read_text(encoding="utf-8") or "(empty)")

            if stdout_path.exists():
                st.subheader("stdout")
                st.code(stdout_path.read_text(encoding="utf-8") or "(empty)")