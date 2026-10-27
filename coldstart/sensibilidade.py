"""Análise de sensibilidade: um parâmetro por vez, em torno do cenário central.

Cada cenário altera um único parâmetro da grade `sensibilidade` do YAML,
mantendo os demais (inclusive a demanda e a compatibilidade calibradas) no
cenário central. Todos os cenários usam as mesmas sementes, de modo que as
diferenças entre cenários também são pareadas.

A margem delta não exige novas simulações: sua sensibilidade é calculada na
análise principal (tabela nao_inferioridade_por_delta).

Cenários já concluídos são reaproveitados se o comando for interrompido e
executado de novo.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analise import CORES, MARCADORES, _estilo_eixos, diferencas_pareadas
from .estatistica import holm, resumo_diferenca, wilcoxon_unilateral
from .experimento import rodar_sementes
from .parametros_sim import ParametrosSimulacao, carregar_parametros
from .saidas import salvar_tabela

SEMENTE_SENSIBILIDADE = 50_000

# chave no YAML -> (atributo em ParametrosSimulacao, rótulo)
PARAMETROS = {
    "taxa_entrada_mensal": ("taxa_entrada", "Taxa de entrada"),
    "churn_mensal": ("churn", "Saída mensal"),
    "gamma": ("gamma", "Viés de posição γ"),
    "tau": ("tau", "Sensibilidade da escolha τ"),
    "u0": ("u0", "Opção externa u₀"),
    "p_review": ("p_review", "Probabilidade de avaliação"),
    "sigma_ruido": ("sigma", "Ruído das avaliações σ"),
    "kappa_priori": ("kappa", "Força da priori κ (C)"),
    "rho": ("rho", "Fração reservada ρ (D)"),
    "janela_novidade_w": ("janela_novidade", "Janela de novidade w (D)"),
}


def posicoes_para_rho(rho: float, K: int, base: tuple) -> tuple:
    """Posições reservadas para outra fração rho, espalhadas entre a primeira e a última da base."""
    n = math.ceil(rho * K)
    if n == len(base):
        return tuple(base)
    if n <= 1:
        return (base[0],)
    return tuple(sorted({int(round(x)) for x in np.linspace(base[0], base[-1], n)}))


def cenarios(config: dict, central: ParametrosSimulacao) -> list[dict]:
    """Lista de cenários: o central e um por valor alternativo de cada parâmetro."""
    lista = [{"nome": "central", "parametro": "central", "valor": None, "p": central}]
    for chave, valores in config["sensibilidade"].items():
        if chave not in PARAMETROS:
            continue
        atributo, _ = PARAMETROS[chave]
        atual = getattr(central, atributo)
        for valor in valores:
            if np.isclose(float(valor), float(atual)):
                continue
            alteracoes = {atributo: type(atual)(valor)}
            if atributo == "rho":
                alteracoes["posicoes_reservadas"] = posicoes_para_rho(valor, central.K,
                                                                      central.posicoes_reservadas)
            lista.append({"nome": f"{chave}={valor}", "parametro": chave, "valor": valor,
                          "p": central.com(**alteracoes)})
    return lista


def executar_sensibilidade(config: dict, pasta_resultados: str | Path, n_sementes: int | None,
                           processos: int) -> pd.DataFrame:
    pasta = Path(pasta_resultados)
    saida = pasta / "sensibilidade"
    (saida / "cenarios").mkdir(parents=True, exist_ok=True)
    central = carregar_parametros(config, pasta)
    n = n_sementes or int(config["simulacao"]["sensibilidade"]["sementes"])
    sementes = [SEMENTE_SENSIBILIDADE + i for i in range(n)]
    lista = cenarios(config, central)
    for i, cenario in enumerate(lista, start=1):
        arquivo = saida / "cenarios" / f"{cenario['nome']}.csv"
        if arquivo.exists() and pd.read_csv(arquivo)["semente"].nunique() == n:
            print(f"  [{i}/{len(lista)}] {cenario['nome']}: já concluído")
            continue
        print(f"  [{i}/{len(lista)}] {cenario['nome']} ...", flush=True)
        df = pd.DataFrame(rodar_sementes(cenario["p"], sementes, processos))
        df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(arquivo, index=False)
    return resumir_sensibilidade(config, pasta, lista)


def resumir_sensibilidade(config: dict, pasta: Path, lista: list[dict]) -> pd.DataFrame:
    saida = pasta / "sensibilidade"
    primarias = [tuple(c) for c in config["protocolo"]["comparacoes_primarias"]]
    delta = float(config["protocolo"]["delta_nao_inferioridade"])
    linhas = []
    for cenario in lista:
        df = pd.read_csv(saida / "cenarios" / f"{cenario['nome']}.csv")
        p_rmst = [wilcoxon_unilateral(diferencas_pareadas(df, "rmst_alta", a, b), "menor")
                  for a, b in primarias]
        p_holm = holm(p_rmst)
        for (a, b), ph in zip(primarias, p_holm):
            rmst = resumo_diferenca(diferencas_pareadas(df, "rmst_alta", a, b))
            ndcg = resumo_diferenca(diferencas_pareadas(df, "ndcg10", a, b))
            linhas.append({
                "cenario": cenario["nome"], "parametro": cenario["parametro"],
                "valor": cenario["valor"], "comparacao": f"{a} vs {b}",
                "delta_rmst_alta": rmst["diferenca_media"], "rmst_ic95_inf": rmst["ic95_inf"],
                "rmst_ic95_sup": rmst["ic95_sup"], "p_holm_H1": ph,
                "H1_sustentada": bool(ph < 0.05 and rmst["diferenca_media"] < 0),
                "delta_ndcg10": ndcg["diferenca_media"], "ndcg_ic90_inf": ndcg["ic90_inf"],
                "H2_sustentada": bool(ndcg["ic90_inf"] > -delta),
            })
    tabela = pd.DataFrame(linhas)
    salvar_tabela(tabela, saida, "sensibilidade",
                  "Sensibilidade das comparações primárias (um parâmetro por vez)",
                  "tab:sensibilidade", casas=4)
    figura_sensibilidade(tabela, saida / "fig_sensibilidade.pdf")
    figura_sensibilidade_artigo(tabela, saida / "fig_sensibilidade_artigo.pdf")
    return tabela


CONGESTAO = {"taxa_entrada_mensal", "churn_mensal"}


def rotulo_cenario(nome: str) -> str:
    """Rótulo legível: 'churn_mensal=0.01' -> 'Saída mensal = 0,01'."""
    if nome == "central":
        return "Cenário central"
    chave, valor = nome.split("=", 1)
    rotulo = PARAMETROS.get(chave, (None, chave))[1]
    return f"{rotulo} = {valor.replace('.', ',')}"


def figura_sensibilidade(tabela: pd.DataFrame, destino: Path) -> None:
    """Diferença de RMST (P - B) com IC 95%, em dois painéis com eixos próprios.

    Painel superior: congestão do mercado (taxa de entrada e saída), que altera a
    relação entre oferta e demanda e produz efeitos de outra ordem de grandeza.
    Painel inferior: demais parâmetros. O cenário central aparece nos dois.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tabela = tabela.copy()
    tabela["painel"] = np.where(tabela["parametro"].isin(CONGESTAO), "congestao", "demais")
    central = tabela[tabela["cenario"] == "central"]
    paineis = [
        ("Congestão do mercado (entrada e saída)",
         pd.concat([central, tabela[tabela["painel"] == "congestao"]])),
        ("Demais parâmetros", pd.concat([central, tabela[(tabela["painel"] == "demais")
                                                         & (tabela["cenario"] != "central")]])),
    ]
    tamanhos = [max(1, parte["cenario"].nunique()) for _, parte in paineis]
    fig, eixos = plt.subplots(2, 1, figsize=(6.0, 0.27 * sum(tamanhos) + 1.8),
                              gridspec_kw={"height_ratios": tamanhos})
    comparacoes = sorted(tabela["comparacao"].unique())
    for ax, (titulo, parte) in zip(eixos, paineis):
        ordem = list(dict.fromkeys(parte["cenario"]))
        y = np.arange(len(ordem))[::-1]
        posicao = dict(zip(ordem, y))
        for deslocamento, comparacao in zip((0.15, -0.15), comparacoes):
            politica = comparacao.split()[0]
            linhas = parte[parte["comparacao"] == comparacao]
            yy = np.array([posicao[c] for c in linhas["cenario"]]) + deslocamento
            ax.errorbar(linhas["delta_rmst_alta"], yy,
                        xerr=[linhas["delta_rmst_alta"] - linhas["rmst_ic95_inf"],
                              linhas["rmst_ic95_sup"] - linhas["delta_rmst_alta"]],
                        fmt=MARCADORES[politica], color=CORES[politica], markersize=5,
                        elinewidth=1.2, capsize=2, label=comparacao)
        ax.axvline(0, color="#55554f", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([rotulo_cenario(c) for c in ordem], fontsize=7)
        ax.set_title(titulo, fontsize=8, loc="left", color="#2b2b28")
        _estilo_eixos(ax)
    eixos[-1].set_xlabel("Δ RMST do TTFC, novatos de alta qualidade (rodadas; < 0 favorece o novato)",
                         fontsize=8)
    eixos[0].legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def figura_sensibilidade_artigo(tabela: pd.DataFrame, destino: Path) -> None:
    """Versão de um painel para o artigo: só os parâmetros de comportamento e das
    políticas (sem entrada e saída, que o texto relata à parte)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parte = tabela[~tabela["parametro"].isin(CONGESTAO)]
    parte = pd.concat([parte[parte["cenario"] == "central"], parte[parte["cenario"] != "central"]])
    ordem = list(dict.fromkeys(parte["cenario"]))
    y = np.arange(len(ordem))[::-1]
    posicao = dict(zip(ordem, y))
    fig, ax = plt.subplots(figsize=(6.0, 0.22 * len(ordem) + 1.1))
    for deslocamento, comparacao in zip((0.15, -0.15), sorted(parte["comparacao"].unique())):
        politica = comparacao.split()[0]
        linhas = parte[parte["comparacao"] == comparacao]
        yy = np.array([posicao[c] for c in linhas["cenario"]]) + deslocamento
        ax.errorbar(linhas["delta_rmst_alta"], yy,
                    xerr=[linhas["delta_rmst_alta"] - linhas["rmst_ic95_inf"],
                          linhas["rmst_ic95_sup"] - linhas["delta_rmst_alta"]],
                    fmt=MARCADORES[politica], color=CORES[politica], markersize=4.5,
                    elinewidth=1.1, capsize=2, label=comparacao)
    ax.axvline(0, color="#55554f", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([rotulo_cenario(c) for c in ordem], fontsize=7)
    _estilo_eixos(ax)
    ax.set_xlabel("Δ RMST do TTFC, novatos de alta qualidade (meses; < 0 favorece o novato)",
                  fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
