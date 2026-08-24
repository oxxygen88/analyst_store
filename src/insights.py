from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .analytics import TargetStatus
from .utils import pct, rupiah


def management_alerts(target: TargetStatus | None, pareto: pd.DataFrame, inventory: pd.DataFrame) -> List[str]:
    alerts: List[str] = []
    if target:
        if target.projected_gap > 0:
            alerts.append(
                f"Target berisiko tidak tercapai: proyeksi akhir bulan {rupiah(target.projected_month_end)}; projected gap {rupiah(target.projected_gap)}."
            )
        else:
            alerts.append(f"Pace saat ini memproyeksikan target tercapai: {rupiah(target.projected_month_end)}.")
        if target.remaining_days > 0:
            alerts.append(
                f"Kebutuhan sales rata-rata {rupiah(target.required_daily_sales)}/hari untuk {target.remaining_days} hari tersisa."
            )
    if not pareto.empty:
        core = pareto[pareto["pareto_group"].eq("CORE_20")]
        at_risk = core[core["current_stock"].le(0)] if "current_stock" in core.columns else pd.DataFrame()
        if not at_risk.empty:
            alerts.append(f"{len(at_risk):,} Core Product sedang stockout/negative dan berisiko kehilangan revenue.".replace(",", "."))
        opportunity = pareto[(pareto["pareto_group"].eq("OPPORTUNITY")) & (pareto.get("opportunity_score", 0) >= 70)]
        if not opportunity.empty:
            alerts.append(f"{len(opportunity):,} Opportunity Product memiliki score ≥70 dan layak diprioritaskan untuk sales push.".replace(",", "."))
    if not inventory.empty:
        over = inventory[inventory["inventory_status"].eq("OVERSTOCK")]
        dead = inventory[inventory["inventory_status"].eq("DEAD")]
        neg = inventory[inventory["inventory_status"].eq("NEGATIVE")]
        if not over.empty:
            alerts.append(f"Overstock terdeteksi pada {len(over):,} SKU dengan nilai stok sekitar {rupiah(over['current_stock_value'].clip(lower=0).sum())}.".replace(",", "."))
        if not dead.empty:
            alerts.append(f"Dead Stock terdeteksi pada {len(dead):,} SKU; perlu evaluasi buying/transfer/clearance.".replace(",", "."))
        if not neg.empty:
            alerts.append(f"{len(neg):,} SKU memiliki stock negatif dan harus masuk proses cleanup.".replace(",", "."))
    return alerts[:6]
