"""Cenários de congestão do mercado: baixa, média e alta.

Cada nível altera a entrada e a saída de fornecedores (`congestao.niveis`),
mantendo os demais parâmetros, inclusive a demanda e a compatibilidade
calibradas. Em cada nível, roda-se o experimento pareado com as políticas das
comparações primárias e aplicam-se os mesmos critérios de H1 a H4. A tabela
também informa o tamanho do mercado ao fim da simulação e o número médio de
fornecedores elegíveis por consulta, que medem a congestão de fato.

Níveis concluídos são reaproveitados se o comando for interrompido.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analise import decidir_hipoteses, diferencas_pareadas
from .estatistica import resumo_diferenca
from .experimento import rodar_sementes
from .parametros_sim import carregar_parametros
from .saidas import salvar_tabela

SEMENTE_CONGESTAO = 80_000
ATRIBUTOS = {"taxa_entrada_mensal": "taxa_entrada", "churn_mensal": "churn"}


def executar_congestao(config: dict, pasta_resultados: str | Path, n_sementes: int | None,
                       processos: int) -> pd.DataFrame:
    pasta = Path(pasta_resultados)
    saida = pasta / "congestao"
    (saida / "niveis").mkdir(parents=True, exist_ok=True)
    cfg = config["congestao"]
    central = carregar_parametros(config, pasta)
    primarias = [tuple(c) for c in config["protocolo"]["comparacoes_primarias"]]
    politicas = tuple(sorted({x for par in primarias for x in par}))
    delta = float(config["protocolo"]["delta_nao_inferioridade"])
    n = n_sementes or int(cfg["sementes"])
    sementes = [SEMENTE_CONGESTAO + i for i in range(n)]

    linhas = []
    for nivel, alteracoes in cfg["niveis"].items():
        alteracoes = alteracoes or {}
        p = central.com(**{ATRIBUTOS[k]: float(v) for k, v in alteracoes.items()})
        arquivo = saida / "niveis" / f"{nivel}.csv"
        if arquivo.exists() and pd.read_csv(arquivo)["semente"].nunique() == n:
            print(f"  {nivel}: já concluído")
        else:
            print(f"  {nivel}: entrada {p.taxa_entrada:.3%}, saída {p.churn:.1%}, {n} sementes ...",
                  flush=True)
            df = pd.DataFrame(rodar_sementes(p, sementes, processos, politicas))
            df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(arquivo, index=False)
        df = pd.read_csv(arquivo)
        hipoteses = decidir_hipoteses(df, primarias, delta)
        for _, h in hipoteses.iterrows():
            a, b = h["comparacao"].split(" vs ")
            r = resumo_diferenca(diferencas_pareadas(df, h["metrica"], a, b))
            linhas.append({
                "nivel": nivel, "taxa_entrada": p.taxa_entrada, "churn": p.churn,
                "fornecedores_ativos_fim": float(df["fornecedores_ativos_fim"].mean()),
                "elegiveis_por_consulta": float(df["elegiveis_por_consulta"].mean()),
                "hipotese": h["hipotese"], "comparacao": h["comparacao"], "metrica": h["metrica"],
                "diferenca_media": r["diferenca_media"], "ic95_inf": r["ic95_inf"],
                "ic95_sup": r["ic95_sup"], "ic90_inf": r["ic90_inf"],
                "sustentada": bool(h["sustentada"]),
            })
    tabela = pd.DataFrame(linhas)
    salvar_tabela(tabela, saida, "congestao",
                  "Hipóteses H1 a H4 em três níveis de congestão do mercado (entrada e saída)",
                  "tab:congestao", casas=4)
    return tabela
