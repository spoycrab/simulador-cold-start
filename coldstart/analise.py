"""Análise do experimento principal: tabelas, decisões das hipóteses e figuras.

Entradas (geradas por `python -m coldstart experimento`):
  resultados/experimento/metricas_por_semente.csv
  resultados/experimento/km_alta.csv
Saídas em resultados/analise/: tabelas .csv e .tex e figuras .pdf.

Análise adicional (revisão do orientador): com a política B2 no experimento, a
diferença C - B é decomposta, semente a semente, em (C - B2) + (B2 - B), isto é,
efeito da exploração mais efeito do sinal. As comparações adicionais são
descritivas (IC por bootstrap pareado e Wilcoxon bilateral), sem decisão de
hipótese; H1 a H4 continuam usando apenas C vs B e D vs B.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .estatistica import holm, resumo_diferenca, wilcoxon_unilateral
from .saidas import salvar_tabela

# Hipóteses pré-registradas: métrica, direção esperada (P - B) e tipo de teste
HIPOTESES = {
    "H1": {"metrica": "rmst_alta", "direcao": "menor", "tipo": "superioridade",
           "descricao": "RMST do TTFC de novatos de alta qualidade menor que em B"},
    "H2": {"metrica": "ndcg10", "direcao": None, "tipo": "nao_inferioridade",
           "descricao": "NDCG@10 não inferior a B (margem delta)"},
    "H3": {"metrica": "descoberta_alta", "direcao": "maior", "tipo": "superioridade",
           "descricao": "maior descoberta de novatos de alta qualidade que em B"},
    "H4": {"metrica": "gini_exposicao", "direcao": "menor", "tipo": "superioridade",
           "descricao": "menor concentração da exposição (Gini) que em B"},
}

METRICAS = {
    "rmst_alta": "RMST do TTFC, novatos de alta qualidade (rodadas)",
    "rmst_todos": "RMST do TTFC, todos os novatos (rodadas)",
    "p_contratacao_ate_h_alta": "P(contratação até h), alta qualidade",
    "descoberta_alta": "Descoberta de novatos de alta qualidade",
    "ndcg10": "NDCG@10",
    "precisao10": "Precision@10",
    "gini_exposicao": "Gini da exposição normalizada",
    "arrependimento_por_contratacao": "Arrependimento por contratação",
    "conversao": "Conversão (contratações por consulta)",
}

POLITICAS_ROTULO = {"A": "A: média simples", "B": "B: reputação bayesiana",
                    "B2": "B2: média posterior Beta", "C": "C: Thompson Sampling",
                    "D": "D: cota temporal"}
# Paleta categórica de referência (ordem fixa) + estilos para impressão em preto e branco
CORES = {"A": "#2a78d6", "B": "#eb6834", "B2": "#7a5bd1", "C": "#1baf7a", "D": "#eda100"}
ESTILOS = {"A": ":", "B": "--", "B2": (0, (5, 1, 1, 1)), "C": "-", "D": "-."}
MARCADORES = {"A": "s", "B": "o", "B2": "v", "C": "^", "D": "D"}
POLITICAS_FIGURA_KM = ("A", "B", "C", "D")   # a Figura 1 do artigo não inclui B2

# Decomposição C - B = (C - B2) + (B2 - B)
METRICAS_DECOMPOSICAO = {
    "rmst_alta": "RMST, alta qualidade (meses)",
    "rmst_todos": "RMST, todos os novatos (meses)",
    "descoberta_alta": "Descoberta, alta qualidade",
    "ndcg10": "NDCG@10",
    "gini_exposicao": "Gini da exposição",
}


def diferencas_pareadas(df: pd.DataFrame, metrica: str, a: str, b: str) -> np.ndarray:
    tabela = df.pivot(index="semente", columns="politica", values=metrica)
    return (tabela[a] - tabela[b]).dropna().to_numpy()


def tabela_por_politica(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for politica, grupo in df.groupby("politica"):
        for metrica, nome in METRICAS.items():
            x = grupo[metrica].to_numpy(float)
            meia = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
            linhas.append({"Política": politica, "Métrica": nome, "Média": x.mean(),
                           "IC95 inf.": x.mean() - meia, "IC95 sup.": x.mean() + meia})
    return pd.DataFrame(linhas)


def tabela_comparacoes(df: pd.DataFrame, comparacoes: list[tuple[str, str]]) -> pd.DataFrame:
    """Diferenças pareadas (P - B) para todas as métricas e comparações."""
    linhas = []
    for metrica, nome in METRICAS.items():
        for a, b in comparacoes:
            d = diferencas_pareadas(df, metrica, a, b)
            resumo = resumo_diferenca(d, semente=len(linhas))
            from scipy import stats
            p_bilateral = (float(stats.wilcoxon(d).pvalue)
                           if len(d) > 0 and not np.all(d == 0) else 1.0)
            linhas.append({"metrica": metrica, "descricao": nome, "comparacao": f"{a} vs {b}",
                           **resumo, "p_wilcoxon_bilateral": p_bilateral})
    tabela = pd.DataFrame(linhas)
    # Holm sobre as comparações de cada métrica (família = métrica)
    tabela["p_holm_bilateral"] = np.nan
    for metrica, grupo in tabela.groupby("metrica"):
        tabela.loc[grupo.index, "p_holm_bilateral"] = holm(grupo["p_wilcoxon_bilateral"].tolist())
    return tabela


def decidir_hipoteses(df: pd.DataFrame, primarias: list[tuple[str, str]], delta: float) -> pd.DataFrame:
    """Aplica os critérios pré-registrados de H1 a H4 às comparações primárias."""
    linhas = []
    for h, info in HIPOTESES.items():
        resumos = [(a, b, diferencas_pareadas(df, info["metrica"], a, b)) for a, b in primarias]
        if info["tipo"] == "superioridade":
            p = [wilcoxon_unilateral(d, info["direcao"]) for _, _, d in resumos]
            p_ajustado = holm(p)
            for (a, b, d), pv, pa in zip(resumos, p, p_ajustado):
                media = float(d.mean())
                na_direcao = media < 0 if info["direcao"] == "menor" else media > 0
                linhas.append({"hipotese": h, "comparacao": f"{a} vs {b}", "metrica": info["metrica"],
                               "diferenca_media": media, "p_unilateral": pv, "p_holm": pa,
                               "criterio": "p de Holm < 0,05 na direção da hipótese",
                               "sustentada": bool(pa < 0.05 and na_direcao)})
        else:
            for a, b, d in resumos:
                r = resumo_diferenca(d)
                linhas.append({"hipotese": h, "comparacao": f"{a} vs {b}", "metrica": info["metrica"],
                               "diferenca_media": r["diferenca_media"], "ic90_inf": r["ic90_inf"],
                               "criterio": f"limite inferior do IC 90% > -{delta}",
                               "sustentada": bool(r["ic90_inf"] > -delta)})
    return pd.DataFrame(linhas)


def nao_inferioridade_por_delta(df: pd.DataFrame, primarias, deltas) -> pd.DataFrame:
    """Sensibilidade da decisão de H2 à margem delta (sem novas simulações)."""
    linhas = []
    for a, b in primarias:
        r = resumo_diferenca(diferencas_pareadas(df, "ndcg10", a, b))
        for delta in deltas:
            linhas.append({"comparacao": f"{a} vs {b}", "delta": delta, "ic90_inf": r["ic90_inf"],
                           "nao_inferior": bool(r["ic90_inf"] > -delta)})
    return pd.DataFrame(linhas)


def tabela_decomposicao(df: pd.DataFrame) -> pd.DataFrame:
    """C - B = (C - B2) + (B2 - B), por métrica, com IC 95% por bootstrap pareado.

    A soma vale exatamente em cada semente; os intervalos são calculados para
    cada parcela separadamente.
    """
    linhas = []
    for i, (metrica, nome) in enumerate(METRICAS_DECOMPOSICAO.items()):
        partes = {}
        for rotulo, (a, b) in {"C - B (total)": ("C", "B"), "C - B2 (exploração)": ("C", "B2"),
                               "B2 - B (sinal)": ("B2", "B")}.items():
            r = resumo_diferenca(diferencas_pareadas(df, metrica, a, b), semente=i)
            partes[rotulo] = r
            linhas.append({"metrica": metrica, "descricao": nome, "parcela": rotulo,
                           "diferenca_media": r["diferenca_media"], "ic95_inf": r["ic95_inf"],
                           "ic95_sup": r["ic95_sup"]})
        total = partes["C - B (total)"]["diferenca_media"]
        soma = partes["C - B2 (exploração)"]["diferenca_media"] + partes["B2 - B (sinal)"]["diferenca_media"]
        assert np.isclose(total, soma), (metrica, total, soma)
    tabela = pd.DataFrame(linhas)
    totais = tabela[tabela["parcela"] == "C - B (total)"].set_index("metrica")["diferenca_media"]
    tabela["fracao_do_total"] = [
        r["diferenca_media"] / totais[r["metrica"]] if totais[r["metrica"]] != 0 else np.nan
        for _, r in tabela.iterrows()]
    return tabela


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _estilo_eixos(ax) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#9a9a96")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def figura_kaplan_meier(km: pd.DataFrame, destino: Path, horizonte: int) -> None:
    """Curvas médias de P(ainda não contratado) para novatos de alta qualidade."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for politica in [p for p in POLITICAS_FIGURA_KM if p in set(km["politica"])]:
        curvas = km[km["politica"] == politica].pivot(index="semente", columns="t", values="S")
        media = curvas.mean().to_numpy()
        meia = 1.96 * curvas.std(ddof=1).to_numpy() / np.sqrt(len(curvas))
        t = curvas.columns.to_numpy()
        ax.step(t, media, where="post", color=CORES[politica], linestyle=ESTILOS[politica],
                linewidth=2, label=POLITICAS_ROTULO[politica])
        ax.fill_between(t, media - meia, media + meia, step="post", color=CORES[politica],
                        alpha=0.15, linewidth=0)
    ax.set_xlim(0, horizonte + 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Rodadas (meses) desde a entrada", fontsize=9, color="#2b2b28")
    ax.set_ylabel("Proporção ainda não contratada", fontsize=9, color="#2b2b28")
    _estilo_eixos(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def figura_fronteira(comparacoes: pd.DataFrame, destino: Path, delta: float) -> None:
    """Diferença de NDCG@10 (cliente) contra diferença de RMST (prestador), em relação a B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ndcg = comparacoes[comparacoes["metrica"] == "ndcg10"].set_index("comparacao")
    rmst = comparacoes[comparacoes["metrica"] == "rmst_alta"].set_index("comparacao")
    for comparacao in ndcg.index:
        if not comparacao.endswith(" vs B"):
            continue                     # a figura mostra só as comparações com B
        politica = comparacao.split()[0]
        x, y = ndcg.loc[comparacao], rmst.loc[comparacao]
        ax.errorbar(x["diferenca_media"], y["diferenca_media"],
                    xerr=[[x["diferenca_media"] - x["ic95_inf"]], [x["ic95_sup"] - x["diferenca_media"]]],
                    yerr=[[y["diferenca_media"] - y["ic95_inf"]], [y["ic95_sup"] - y["diferenca_media"]]],
                    fmt=MARCADORES[politica], color=CORES[politica], markersize=7, elinewidth=1.5,
                    capsize=3, markeredgecolor="#fcfcfb", label=POLITICAS_ROTULO[politica])
        ax.annotate(politica, (x["diferenca_media"], y["diferenca_media"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color="#2b2b28")
    ax.axvline(-delta, color="#55554f", linewidth=1, linestyle="--")
    ax.annotate(f"margem -δ = -{delta}", (-delta, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=7, color="#55554f")
    ax.axhline(0, color="#9a9a96", linewidth=0.8)
    ax.axvline(0, color="#9a9a96", linewidth=0.8)
    ax.set_xlabel("Δ NDCG@10 em relação a B (maior = melhor para o cliente)", fontsize=9)
    ax.set_ylabel("Δ RMST do TTFC em relação a B\n(menor = melhor para o novato)", fontsize=9)
    _estilo_eixos(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def executar_analise(config: dict, pasta_resultados: str | Path) -> dict:
    pasta = Path(pasta_resultados)
    entrada = pasta / "experimento"
    arquivo = entrada / "metricas_por_semente.csv"
    if not arquivo.exists():
        raise FileNotFoundError("Experimento não encontrado. Rode antes: python -m coldstart experimento")
    df = pd.read_csv(arquivo)
    km = pd.read_csv(entrada / "km_alta.csv")
    parametros = json.loads((entrada / "parametros.json").read_text(encoding="utf-8"))
    saida = pasta / "analise"
    saida.mkdir(parents=True, exist_ok=True)

    protocolo = config["protocolo"]
    primarias = [tuple(c) for c in protocolo["comparacoes_primarias"]]
    presentes = set(df["politica"])
    todas = primarias + ([("A", "B")] if "A" in presentes else [])
    adicionais = [tuple(c) for c in protocolo.get("comparacoes_adicionais", [])
                  if set(c) <= presentes]
    delta = float(protocolo["delta_nao_inferioridade"])

    por_politica = tabela_por_politica(df)
    comparacoes = tabela_comparacoes(df, todas)
    hipoteses = decidir_hipoteses(df, primarias, delta)
    deltas = nao_inferioridade_por_delta(df, primarias, config["sensibilidade"]["delta_nao_inferioridade"])

    salvar_tabela(por_politica, saida, "metricas_por_politica",
                  f"Métricas por política ({df['semente'].nunique()} sementes, média e IC 95%)",
                  "tab:metricas_por_politica", casas=3)
    salvar_tabela(comparacoes, saida, "comparacoes",
                  "Diferenças pareadas em relação a B", "tab:comparacoes", casas=4)
    salvar_tabela(hipoteses, saida, "hipoteses",
                  "Decisão das hipóteses pré-registradas", "tab:hipoteses", casas=4)
    decomposicao = pd.DataFrame()
    if adicionais:
        salvar_tabela(tabela_comparacoes(df, adicionais), saida, "comparacoes_adicionais",
                      "Análises adicionais: C vs B2 (exploração) e B2 vs B (sinal)",
                      "tab:comparacoes_adicionais", casas=4)
    if {"B", "B2", "C"} <= presentes:
        decomposicao = tabela_decomposicao(df)
        salvar_tabela(decomposicao, saida, "decomposicao_b2",
                      "Decomposição C - B = (C - B2) + (B2 - B): exploração e sinal",
                      "tab:decomposicao", casas=4)
    salvar_tabela(deltas, saida, "nao_inferioridade_por_delta",
                  "Não inferioridade do NDCG@10 para diferentes margens", "tab:delta", casas=4)

    figura_kaplan_meier(km, saida / "fig_kaplan_meier_alta.pdf", int(parametros["horizonte_h"]))
    figura_fronteira(comparacoes, saida / "fig_fronteira.pdf", delta)
    return {"hipoteses": hipoteses, "comparacoes": comparacoes, "por_politica": por_politica,
            "decomposicao": decomposicao}
