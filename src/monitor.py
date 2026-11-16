"""Автоматичний моніторинг: порівнює нові дані з еталоном і зберігає звіти
з датою в назві. Запуск з кореня проєкту: python src/monitor.py
Для регулярного запуску додайте цю команду в Планувальник завдань Windows (наприклад, щодня)."""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # щоб працював import app

from app.config import DB_PATH
from app.monitoring import align, build_reports, load_current, load_reference

conn = sqlite3.connect(DB_PATH)
reference, current = align(load_reference(conn)), align(load_current(conn))
conn.close()

stamp = datetime.now().strftime("%Y%m%d_%H%M")
summary = build_reports(reference, current, suffix=f"_{stamp}")

print(f"Еталон: {summary['reference_rows']} записів, поточні: {summary['current_rows']} "
      f"(з відомим класом: {summary['current_with_target']})")
print(f"Ознаки з дрейфом: {summary['drifted_features'] or 'немає'} "
      f"(частка {summary['share_drifted']:.0%})")
for t in summary["target_drift"]:
    print(f"  {t['column']}: {t['method']} = {t['score']} (поріг {t['threshold']}) -> "
          f"{'ДРЕЙФ' if t['drift'] else 'без дрейфу'}")
print("Звіти збережено в reports/:", ", ".join(summary["files"].values()))
if summary["share_drifted"] >= 0.5:
    print("УВАГА: дрейф більш ніж у половині ознак — модель варто перенавчити")
