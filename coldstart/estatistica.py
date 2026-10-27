"""Testes estatísticos pré-registrados para comparações pareadas por semente.

* Intervalos: bootstrap percentil pareado das diferenças por semente.
* Valores-p: Wilcoxon pareado unilateral, na direção da hipótese.
* Multiplicidade: Holm, dentro de cada hipótese (sobre as comparações primárias).
* Não inferioridade (H2): sustentada se o limite inferior do IC 90% bilateral
  (equivalente a um teste unilateral com alfa = 5%) de Delta = NDCG(P) - NDCG(B)
  for maior que -delta.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def ic_bootstrap(diferencas: np.ndarray, nivel: float = 0.95, n_boot: int = 10_000,
                 semente: int = 0) -> tuple[float, float]:
    """IC percentil da média das diferenças pareadas."""
    d = np.asarray(diferencas, dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semente)
    medias = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    alfa = (1 - nivel) / 2
    return float(np.quantile(medias, alfa)), float(np.quantile(medias, 1 - alfa))


def wilcoxon_unilateral(diferencas: np.ndarray, direcao: str) -> float:
    """Valor-p do Wilcoxon pareado; direcao = 'menor' (H: d < 0) ou 'maior' (H: d > 0)."""
    d = np.asarray(diferencas, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    alternativa = "less" if direcao == "menor" else "greater"
    return float(stats.wilcoxon(d, alternative=alternativa, zero_method="wilcox").pvalue)


def holm(valores_p: list[float]) -> list[float]:
    """Correção de Holm-Bonferroni (valores-p ajustados, na ordem original)."""
    p = np.asarray(valores_p, dtype=float)
    m = len(p)
    ordem = np.argsort(p)
    ajustados = np.empty(m)
    acumulado = 0.0
    for posicao, indice in enumerate(ordem):
        acumulado = max(acumulado, (m - posicao) * p[indice])
        ajustados[indice] = min(1.0, acumulado)
    return [float(x) for x in ajustados]


def resumo_diferenca(diferencas: np.ndarray, semente: int = 0) -> dict:
    """Média, ICs 95% e 90%, tamanho de efeito d_z e erro Monte Carlo."""
    d = np.asarray(diferencas, dtype=float)
    n = len(d)
    dp = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    ic95 = ic_bootstrap(d, 0.95, semente=semente)
    ic90 = ic_bootstrap(d, 0.90, semente=semente)
    return {
        "n": n, "diferenca_media": float(d.mean()),
        "ic95_inf": ic95[0], "ic95_sup": ic95[1],
        "ic90_inf": ic90[0], "ic90_sup": ic90[1],
        "d_z": float(d.mean() / dp) if dp and dp > 0 else float("nan"),
        "erro_monte_carlo": dp / np.sqrt(n) if n > 1 else float("nan"),
    }
