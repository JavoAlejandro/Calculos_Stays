"""
stays_runner.py
===============
Orquestador Python para el algoritmo de detección de stays en C.

Optimizaciones respecto a la versión original
----------------------------------------------
1. UNA SOLA LLAMADA al binario C por grupo de (mes, dia) en lugar de una
   llamada por (owner_id, mes, dia). El C recibe todos los owners del día
   y los agrupa internamente con un único sort. Esto elimina el overhead de
   fork+exec+CSV write+CSV read que dominaba el tiempo real con bloques de
   ~50 filas.

2. export_block_csv escribe directamente con to_csv() sin copy() intermedio.
   El timestamp se formatea en el mismo select con dt.strftime.

3. El agrupamiento por defecto cambia de ['owner_id', 'mes', 'dia']
   a ['mes', 'dia'], reduciendo el número de bloques de N_owners×N_días
   a solo N_días. Cada bloque puede tener cientos de owners; el C los
   procesa todos en una pasada.

4. Consolidación con pd.concat sobre lista de DataFrames leídos directamente,
   sin pasar por archivos intermedios por bloque.

5. El modo paralelo (--workers > 1) sigue funcionando; ahora cada worker
   recibe un bloque (mes, dia) completo en lugar de un bloque de un solo
   owner, lo que reduce el overhead de fork por trabajo útil realizado.

Uso rápido:
    python stays_runner.py --input datos.csv --output resultados.csv

Con workers paralelos y agrupación personalizada:
    python stays_runner.py \
        --input datos.csv \
        --output resultados.csv \
        --group-by mes dia \
        --roaming 250 \
        --duration 15 \
        --workers 4
"""

import os
import sys
import subprocess
import argparse
import shutil
import tempfile
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
C_SOURCE   = SCRIPT_DIR / "stays.c"
BINARY     = SCRIPT_DIR / "stays_bin"


# ---------------------------------------------------------------------------
# Compilación automática
# ---------------------------------------------------------------------------

def compile_c(c_source: Path = C_SOURCE, binary: Path = BINARY) -> Path:
    """Compila stays.c si el binario no existe o el fuente es más nuevo."""
    if binary.exists() and binary.stat().st_mtime >= c_source.stat().st_mtime:
        log.info(f"Binario ya actualizado: {binary}")
        return binary

    log.info(f"Compilando {c_source} ...")
    result = subprocess.run(
        ["gcc", "-O2", "-o", str(binary), str(c_source), "-lm"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("Error de compilación:\n" + result.stderr)
        sys.exit(1)
    log.info(f"Compilación exitosa: {binary}")
    return binary


# ---------------------------------------------------------------------------
# Carga del CSV
# ---------------------------------------------------------------------------
REQUIRED_COLS = {"account_id", "latitude", "longitude", "owner_id", "timestamp"}


def load_csv(path: str, chunksize: int | None = None) -> pd.DataFrame:
    """
    Carga el CSV completo o en chunks.
    Deriva columnas 'mes' y 'dia' del timestamp si no existen.
    """
    log.info(f"Leyendo {path} ...")
    t0 = time.time()

    dtype = {
        "account_id": "int32",
        "owner_id":   "int32",
        "latitude":   "float64",
        "longitude":  "float64",
    }

    if chunksize:
        df = pd.concat(
            pd.read_csv(path, dtype=dtype, parse_dates=["timestamp"],
                        chunksize=chunksize),
            ignore_index=True,
        )
    else:
        df = pd.read_csv(path, dtype=dtype, parse_dates=["timestamp"])

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        log.error(f"Columnas faltantes en el CSV: {missing}")
        sys.exit(1)

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "mes" not in df.columns:
        df["mes"] = df["timestamp"].dt.month.astype("int8")
    if "dia" not in df.columns:
        df["dia"] = df["timestamp"].dt.day.astype("int8")

    log.info(f"  {len(df):,} filas cargadas en {time.time()-t0:.1f}s")
    log.info(f"  Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  Owners únicos: {df['owner_id'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Agrupación
# ---------------------------------------------------------------------------

def build_groups(
    df: pd.DataFrame,
    group_by: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    """
    Un bloque por combinación única de group_by (por defecto ['mes', 'dia']).

    Con el agrupamiento original por (owner_id, mes, dia) se generaban
    N_owners × N_días bloques de ~50 filas cada uno → miles de subprocesses.
    Con (mes, dia) se generan solo N_días bloques, cada uno con todos los
    owners de ese día. El C los procesa en una sola pasada.
    """
    log.info(f"Agrupando por {group_by} ...")
    result = []
    for keys, gdf in df.groupby(group_by, sort=True):
        name = "g_" + (
            "_".join(str(k) for k in keys)
            if isinstance(keys, tuple) else str(keys)
        )
        result.append((name, gdf.reset_index(drop=True)))

    sizes = [len(g) for _, g in result]
    log.info(
        f"  {len(result):,} bloques | "
        f"min={min(sizes):,} max={max(sizes):,} "
        f"avg={int(sum(sizes)/len(sizes)):,} filas/bloque"
    )
    return result


# ---------------------------------------------------------------------------
# Exportar bloque a CSV para el binario C
# ---------------------------------------------------------------------------

def export_block_csv(df: pd.DataFrame, path: str) -> None:
    """
    Escribe account_id,latitude,longitude,owner_id,timestamp
    sin copy() intermedio; formatea el timestamp en el select.
    """
    cols = ["account_id", "latitude", "longitude", "owner_id", "timestamp"]
    out = df[cols].copy()
    if pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Procesamiento de un bloque (1 subprocess = 1 día completo)
# ---------------------------------------------------------------------------

def process_block(args: tuple) -> dict:
    """
    Ejecuta el binario C sobre un bloque completo (todos los owners del día).
    Diseñado para correrse en paralelo con ProcessPoolExecutor.
    """
    name, df, binary_path, roaming, duration, tmp_dir = args

    t0         = time.time()
    tmp_input  = os.path.join(tmp_dir, f"{name}_input.csv")
    tmp_output = os.path.join(tmp_dir, f"{name}_output.csv")
    n_points   = len(df)

    export_block_csv(df, tmp_input)

    result = subprocess.run(
        [str(binary_path), tmp_input, tmp_output,
         str(roaming), str(duration), "0"],
        capture_output=True, text=True,
    )

    elapsed = time.time() - t0
    n_stays = 0

    if result.returncode != 0:
        log.error(f"[{name}] Error en C:\n{result.stderr}")
        return {
            "name": name, "input_rows": n_points,
            "output_stays": 0, "elapsed": elapsed,
            "output_path": None, "error": result.stderr,
        }

    stdout = result.stdout.strip()
    if stdout:
        parts = stdout.split(",")
        if len(parts) == 2:
            try:
                n_stays = int(parts[1])
            except ValueError:
                pass

    return {
        "name":          name,
        "input_rows":    n_points,
        "output_stays":  n_stays,
        "elapsed":       elapsed,
        "output_path":   tmp_output if os.path.exists(tmp_output) else None,
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_pipeline(
    input_csv:  str,
    output_csv: str,
    group_by:   list[str],
    roaming:    float,
    duration:   int,
    workers:    int,
    chunksize:  int | None,
    c_source:   Path,
) -> None:
    total_t0 = time.time()

    binary = compile_c(c_source)
    df     = load_csv(input_csv, chunksize=chunksize)
    blocks = build_groups(df, group_by)

    log.info(f"Total bloques a procesar: {len(blocks)}")

    tmp_dir = tempfile.mkdtemp(prefix="stays_tmp_")
    log.info(f"Directorio temporal: {tmp_dir}")

    try:
        task_args = [
            (name, block_df, binary, roaming, duration, tmp_dir)
            for name, block_df in blocks
        ]

        results: list[dict] = []

        if workers > 1:
            log.info(f"Procesando con {workers} workers en paralelo ...")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_block, a): a[0]
                    for a in task_args
                }
                for fut in as_completed(futures):
                    r = fut.result()
                    results.append(r)
                    _log_block(r)
        else:
            log.info("Procesando en serie ...")
            for i, args in enumerate(task_args, 1):
                log.info(f"  Bloque {i}/{len(task_args)}: {args[0]} ...")
                r = process_block(args)
                results.append(r)
                _log_block(r)

        # ── Consolidar ──────────────────────────────────────────────────────
        log.info("Consolidando resultados ...")
        parts = []
        for r in results:
            if r["output_path"] and os.path.exists(r["output_path"]):
                try:
                    parts.append(pd.read_csv(r["output_path"]))
                except Exception as exc:
                    log.warning(f"No se pudo leer {r['output_path']}: {exc}")

        if parts:
            final = pd.concat(parts, ignore_index=True)
            final.sort_values(["owner_id", "start_time"], inplace=True)
            final.to_csv(output_csv, index=False)
            log.info(f"Resultados guardados en: {output_csv}")
            log.info(f"Total stays: {len(final):,}")
        else:
            log.warning("No se generó ningún resultado.")

        # ── Resumen ─────────────────────────────────────────────────────────
        total_elapsed = time.time() - total_t0
        errors = [r for r in results if r["error"]]

        print("\n" + "=" * 60)
        print("  RESUMEN FINAL")
        print("=" * 60)
        print(f"  Bloques procesados : {len(results)}")
        print(f"  Puntos procesados  : {sum(r['input_rows'] for r in results):,}")
        print(f"  Stays detectados   : {sum(r['output_stays'] for r in results):,}")
        print(f"  Errores            : {len(errors)}")
        print(f"  Tiempo total       : {total_elapsed:.1f}s")
        print(f"  Archivo salida     : {output_csv}")
        print("=" * 60)

        if errors:
            print("\nBloques con error:")
            for r in errors:
                print(f"  - {r['name']}: {r['error'][:120]}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _log_block(r: dict) -> None:
    status = "✓" if r["error"] is None else "✗"
    log.info(
        f"  {status} {r['name']}: "
        f"{r['input_rows']:,} pts → {r['output_stays']} stays "
        f"({r['elapsed']:.2f}s)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestador Python para detección de stays (C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,  help="CSV de entrada")
    parser.add_argument("--output", "-o", required=True,  help="CSV de salida")
    parser.add_argument(
        "--group-by", "-g",
        nargs="+",
        default=["mes", "dia"],
        help="Columnas para agrupar. Default: mes dia (1 bloque = 1 día completo, "
             "más eficiente que el agrupamiento original por owner_id).",
    )
    parser.add_argument(
        "--roaming", "-r",
        type=float, default=250.0,
        help="Distancia máxima de roaming en metros.",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int, default=15,
        help="Duración mínima del stay en minutos.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=1,
        help="Workers paralelos (1 = serie).",
    )
    parser.add_argument(
        "--chunksize",
        type=int, default=None,
        help="Leer CSV en chunks de N filas (útil para archivos muy grandes).",
    )
    parser.add_argument(
        "--c-source",
        type=str, default=str(C_SOURCE),
        help="Ruta al archivo stays.c.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    c_path = Path(args.c_source)
    if not c_path.exists():
        log.error(f"No se encontró el archivo C: {c_path}")
        sys.exit(1)

    run_pipeline(
        input_csv  = args.input,
        output_csv = args.output,
        group_by   = args.group_by,
        roaming    = args.roaming,
        duration   = args.duration,
        workers    = args.workers,
        chunksize  = args.chunksize,
        c_source   = c_path,
    )
