"""
Pipeline NBA — ejecución diaria.

Correr ANTES del primer tip-off del día para garantizar
que predicted_at < tip_off_time (requisito de validez del paper).

Uso:
  python run_daily.py          → modo normal (datos de hoy + predicciones)
  python run_daily.py --full   → sincronización completa (primera vez o recovery)
  python run_daily.py --eval   → sólo muestra métricas OOS actuales
"""

import os
import sys
import logging
from datetime import datetime, date

from pipeline.setup_db import create_tables, DB_PATH
from pipeline.ingest_games import fetch_season_game_log, fetch_today_schedule, update_final_scores
from pipeline.ingest_injuries import fetch_injury_report
from pipeline.compute_features import compute_rolling_stats
from pipeline.model import train, predict, evaluate, _latest_version
from pipeline.export import export_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)

RETRAIN_DAY = 0  # lunes (0=lunes … 6=domingo)
MIN_GAMES_TO_TRAIN = 20


def _should_retrain(db_path: str) -> bool:
    """Retrain los lunes o si todavía no existe ningún modelo."""
    if _latest_version(db_path) is None:
        return True
    return date.today().weekday() == RETRAIN_DAY


def run(full_sync: bool = False, eval_only: bool = False):
    start = datetime.now()
    log.info("=" * 50)
    log.info(f"Pipeline NBA  {date.today()}  full={full_sync}")
    log.info("=" * 50)

    if eval_only:
        evaluate()
        return

    # 1. Inicializar base de datos
    create_tables()

    # 2. Actualizar resultados de ayer (también rellena predictions.actual_*)
    update_final_scores()

    # 3. Agenda + líneas O/U de hoy
    fetch_today_schedule()

    # 4. Injury report (ANTES de predecir)
    fetch_injury_report()

    # 5. Sincronización completa del game log (primera vez o recovery)
    if full_sync:
        log.info("Descargando game log completo de la temporada...")
        fetch_season_game_log()

    # 6. Rolling features
    compute_rolling_stats()

    # 7. Reentrenamiento semanal (lunes) o primera vez
    if _should_retrain(DB_PATH):
        log.info("Reentrenando modelo...")
        version_id = train()
        if version_id:
            log.info(f"Modelo listo: {version_id}")
    else:
        version_id = _latest_version(DB_PATH)
        log.info(f"Usando modelo existente: {version_id}")

    # 8. Predicciones para hoy (timestamp pre tip-off)
    if version_id:
        predictions = predict(version_id=version_id)
        if predictions:
            log.info(f"\nPredicciones del día ({len(predictions)} partidos):")
            for p in predictions:
                log.info(f"  {p['home']:>3} {p['pred_home']:.0f} – {p['pred_away']:.0f} {p['away']:<3}  "
                         f"total={p['pred_total']:.0f}")

    # 9. Métricas OOS acumuladas (para seguimiento del paper)
    evaluate()

    # 10. Exportar JSON + push a GitHub (actualiza dashboard en Netlify)
    export_json()
    _git_push()

    elapsed = (datetime.now() - start).seconds
    log.info(f"\nPipeline completado en {elapsed}s")


def _git_push():
    import subprocess
    try:
        subprocess.run(['git', 'add', 'public/data.json'], check=True,
                       cwd=os.path.dirname(__file__) or '.')
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'],
            cwd=os.path.dirname(__file__) or '.'
        )
        if result.returncode != 0:  # hay cambios staged
            from datetime import date
            subprocess.run(
                ['git', 'commit', '-m', f'data: daily update {date.today()}'],
                check=True, cwd=os.path.dirname(__file__) or '.'
            )
            subprocess.run(['git', 'push'], check=True,
                           cwd=os.path.dirname(__file__) or '.')
            log.info("data.json pusheado → Netlify actualizará en ~30s")
        else:
            log.info("Sin cambios en data.json, push omitido.")
    except Exception as e:
        log.warning(f"Git push falló (no crítico): {e}")


if __name__ == '__main__':
    full_sync = '--full' in sys.argv
    eval_only = '--eval' in sys.argv
    run(full_sync=full_sync, eval_only=eval_only)
