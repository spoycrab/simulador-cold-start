"""Análise de sensibilidade global: triagem de Morris (efeitos elementares).

Todos os parâmetros da lista `morris.parametros` variam juntos. Cada trajetória
parte de um ponto sorteado de uma grade com `niveis` níveis por parâmetro (na
escala [0, 1] da faixa de cada um) e altera um parâmetro por vez, em ordem
aleatória, com passo Delta = niveis / (2 (niveis - 1)). A diferença de saída
entre dois pontos consecutivos, dividida pelo passo, é o efeito elementar
daquele parâmetro. Com r trajetórias, cada parâmetro tem r efeitos elementares:

  mu*   = média dos valores absolutos (importância do parâmetro);
  mu    = média com sinal (direção do efeito);
  sigma = desvio-padrão (não linearidade ou interação com outros parâmetros).

Os efeitos elementares estão na unidade da saída por faixa inteira do
parâmetro (de min a max). Saídas: diferenças pareadas por semente, em relação
a B, de RMST (alta qualidade), NDCG@10 e incidência cumulativa da contratação
antes da saída, para C e D. Todos os pontos usam as mesmas sementes.

A demanda e a compatibilidade entram como multiplicadores do valor calibrado.
Trajetórias concluídas são reaproveitadas se o comando for interrompido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analise import _estilo_eixos
from .experimento import mapear, rodar_semente
from .parametros_sim import ParametrosSimulacao, carregar_parametros
from .saidas import salvar_tabela

SEMENTE_MORRIS = 70_000
POLITICAS_MORRIS = ("B", "C", "D")
SAIDAS = {
    "rmst_alta": "Δ RMST, alta qualidade (meses)",
    "ndcg10": "Δ NDCG@10",
    "cif_contratacao_h_alta": "Δ contratação antes da saída, alta qualidade",
}
COMPARACOES = (("C", "B"), ("D", "B"))


def trajetorias(k: int, r: int, niveis: int, semente: int) -> list[np.ndarray]:
    """r trajetórias de Morris, cada uma com k + 1 pontos na grade {0, 1/(p-1), ..., 1}^k."""
    rng = np.random.default_rng(semente)
    passo = niveis / (2 * (niveis - 1))
    grade = np.linspace(0.0, 1.0, niveis)
    lista = []
    for _ in range(r):
        x = rng.choice(grade, size=k)
        pontos = [x.copy()]
        for i in rng.permutation(k):
            x[i] = x[i] + passo if x[i] + passo <= 1 + 1e-12 else x[i] - passo
            pontos.append(x.copy())
        lista.append(np.round(np.array(pontos), 12))
    return lista


def parametros_do_ponto(central: ParametrosSimulacao, especificacao: dict,
                        unidade: np.ndarray) -> ParametrosSimulacao:
    """Converte um ponto em [0, 1]^k nos parâmetros da simulação."""
    alteracoes = {}
    for u, (nome, spec) in zip(unidade, especificacao.items()):
        valor = spec["min"] + u * (spec["max"] - spec["min"])
        atributo = spec["atributo"]
        if spec.get("relativo", False):
            valor = valor * float(getattr(central, atributo))
        atual = getattr(central, atributo)
        alteracoes[atributo] = type(atual)(valor) if not isinstance(atual, (np.ndarray,)) else valor
    return central.com(**alteracoes)


def _saidas_do_ponto(linhas: list[dict]) -> dict:
    """Médias, entre sementes, das diferenças pareadas P - B."""
    df = pd.DataFrame(linhas)
    resultado = {}
    for metrica in SAIDAS:
        tabela = df.pivot(index="semente", columns="politica", values=metrica)
        for a, b in COMPARACOES:
            resultado[f"{metrica}|{a}-{b}"] = float((tabela[a] - tabela[b]).mean())
    return resultado


def executar_morris(config: dict, pasta_resultados: str | Path, n_sementes: int | None,
                    processos: int) -> pd.DataFrame:
    pasta = Path(pasta_resultados)
    saida = pasta / "morris"
    (saida / "trajetorias").mkdir(parents=True, exist_ok=True)
    cfg = config["morris"]
    especificacao = cfg["parametros"]
    nomes = list(especificacao)
    central = carregar_parametros(config, pasta)
    n = n_sementes or int(cfg["sementes"])
    sementes = [SEMENTE_MORRIS + i for i in range(n)]
    lista = trajetorias(len(nomes), int(cfg["trajetorias"]), int(cfg["niveis"]),
                        int(cfg["semente_desenho"]))

    for t, pontos in enumerate(lista, start=1):
        arquivo = saida / "trajetorias" / f"trajetoria_{t:02d}.csv"
        if arquivo.exists() and len(pd.read_csv(arquivo)) == len(pontos):
            print(f"  [{t}/{len(lista)}] trajetória já concluída")
            continue
        print(f"  [{t}/{len(lista)}] {len(pontos)} pontos x {n} sementes ...", flush=True)
        params = [parametros_do_ponto(central, especificacao, u) for u in pontos]
        tarefas = [(p, s, POLITICAS_MORRIS) for p in params for s in sementes]
        resultados = mapear(rodar_semente, tarefas, processos)
        linhas = []
        for j, (u, p) in enumerate(zip(pontos, params)):
            bloco = [l for r in resultados[j * n:(j + 1) * n] for l in r]
            linha = {"trajetoria": t, "passo": j}
            linha.update({f"u_{nome}": float(x) for nome, x in zip(nomes, u)})
            linha.update({nome: float(getattr(p, spec["atributo"]))
                          for nome, spec in especificacao.items()})
            linha.update(_saidas_do_ponto(bloco))
            linhas.append(linha)
        pd.DataFrame(linhas).to_csv(arquivo, index=False)
    return resumir_morris(config, pasta)


def efeitos_elementares(pontos: pd.DataFrame, nomes: list[str], colunas_saida: list[str]) -> pd.DataFrame:
    """Um efeito elementar por parâmetro, trajetória e saída."""
    linhas = []
    for t, grupo in pontos.groupby("trajetoria"):
        grupo = grupo.sort_values("passo")
        u = grupo[[f"u_{nome}" for nome in nomes]].to_numpy()
        y = grupo[colunas_saida].to_numpy()
        for j in range(1, len(grupo)):
            mudou = np.flatnonzero(~np.isclose(u[j], u[j - 1]))
            assert len(mudou) == 1, "cada passo de Morris altera um único parâmetro"
            i = int(mudou[0])
            delta = u[j, i] - u[j - 1, i]
            for c, coluna in enumerate(colunas_saida):
                linhas.append({"trajetoria": t, "parametro": nomes[i], "saida": coluna,
                               "efeito": (y[j, c] - y[j - 1, c]) / delta})
    return pd.DataFrame(linhas)


def indices_morris(efeitos: pd.DataFrame) -> pd.DataFrame:
    """mu*, mu e sigma por parâmetro e saída."""
    tabela = (efeitos.groupby(["saida", "parametro"])["efeito"]
              .agg(mu_estrela=lambda e: float(np.mean(np.abs(e))), mu="mean",
                   sigma=lambda e: float(np.std(e, ddof=1)) if len(e) > 1 else float("nan"),
                   r="count")
              .reset_index())
    return tabela.sort_values(["saida", "mu_estrela"], ascending=[True, False])


def resumir_morris(config: dict, pasta: Path) -> pd.DataFrame:
    saida = pasta / "morris"
    arquivos = sorted((saida / "trajetorias").glob("trajetoria_*.csv"))
    pontos = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    nomes = list(config["morris"]["parametros"])
    colunas_saida = [f"{m}|{a}-{b}" for m in SAIDAS for a, b in COMPARACOES]
    efeitos = efeitos_elementares(pontos, nomes, colunas_saida)
    indices = indices_morris(efeitos)
    pontos.to_csv(saida / "pontos.csv", index=False)
    efeitos.to_csv(saida / "efeitos_elementares.csv", index=False)
    salvar_tabela(indices, saida, "morris",
                  "Triagem de Morris: importância (mu*), direção (mu) e não linearidade ou "
                  "interação (sigma) de cada parâmetro", "tab:morris", casas=4)
    figura_morris(indices, saida / "fig_morris.pdf")
    return indices


def figura_morris(indices: pd.DataFrame, destino: Path) -> None:
    """mu* contra sigma, para Δ RMST e Δ NDCG@10 (C - B e D - B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analise import CORES, MARCADORES
    fig, eixos = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, metrica in zip(eixos, ["rmst_alta", "ndcg10"]):
        for a, b in COMPARACOES:
            parte = indices[indices["saida"] == f"{metrica}|{a}-{b}"]
            ax.scatter(parte["mu_estrela"], parte["sigma"], color=CORES[a], marker=MARCADORES[a],
                       s=28, label=f"{a} − {b}", zorder=3)
            for _, linha in parte.nlargest(4, "mu_estrela").iterrows():
                ax.annotate(linha["parametro"], (linha["mu_estrela"], linha["sigma"]),
                            xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                            color="#2b2b28")
        ax.set_xlabel("μ* (importância)", fontsize=8)
        ax.set_ylabel("σ (não linearidade / interação)", fontsize=8)
        ax.set_title(SAIDAS[metrica], fontsize=9, loc="left", color="#2b2b28")
        _estilo_eixos(ax)
    eixos[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
