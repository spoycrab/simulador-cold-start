"""Execução das sementes: trajetórias de referência e experimento pareado.

As políticas do experimento principal vêm de `simulacao.politicas` (padrão A, B, B2,
C, D). B2 é uma análise adicional pedida na revisão; o piloto e a sensibilidade
continuam com as políticas do protocolo (A, B, C, D).

A política de referência (`simulacao.politica_referencia`, padrão D) representa a
prática atual da plataforma: um ranking por reputação com impulso a novos anúncios,
como o descrito por Haldar et al. (2020) para o Airbnb. Ela é usada na calibração,
na validação e no aquecimento. A política B passa a ser o contrafactual de
reputação pura.

Desenho de cada semente:
  1. aquecimento de `aquecimento` rodadas sob a política de referência, a partir da
     amostra real;
  2. o estado ao fim do aquecimento é copiado e cada política roda
     as T rodadas seguintes com os mesmos fluxos exógenos;
  3. as métricas são coletadas apenas nas T rodadas da fase de avaliação.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import metricas, motor
from .fluxos import Fluxos
from .parametros_sim import ParametrosSimulacao
from .politicas import POLITICAS_PROTOCOLO


def numero_de_processos(configurado: int | None) -> int:
    if configurado and configurado > 0:
        return int(configurado)
    return max(1, (os.cpu_count() or 2) - 1)


def mapear(funcao, tarefas: list, processos: int) -> list:
    """Executa em paralelo, preservando a ordem das tarefas."""
    if processos <= 1 or len(tarefas) <= 1:
        return [funcao(t) for t in tarefas]
    with ProcessPoolExecutor(max_workers=min(processos, len(tarefas))) as executor:
        return list(executor.map(funcao, tarefas))


def momentos_referencia(args: tuple) -> dict:
    """Trajetória completa sob a política de referência, para calibração e validação."""
    p, semente = args
    est = motor.simular(p, Fluxos(semente, p), p.politica_referencia)
    return metricas.momentos_simulados(est, p)


def rodar_semente(args: tuple) -> list[dict]:
    """Aquecimento sob a referência e ramificação nas políticas; uma linha por política."""
    p, semente, politicas = args
    fluxos = Fluxos(semente, p)
    aquecido = motor.simular(p, fluxos, p.politica_referencia, fim=p.aquecimento)
    linhas = []
    for politica in politicas:
        est = aquecido.copia()
        est.iniciar_coleta()
        motor.simular(p, fluxos, politica, est=est, inicio=p.aquecimento, coletar=True)
        linha = {"semente": semente, "politica": politica}
        linha.update(metricas.metricas_execucao(est, p))
        linha["_km_alta"] = metricas.curva_km_alta(est, p).tolist()
        linhas.append(linha)
    return linhas


def rodar_sementes(p: ParametrosSimulacao, sementes, processos: int,
                   politicas=POLITICAS_PROTOCOLO) -> list[dict]:
    resultados = mapear(rodar_semente, [(p, s, tuple(politicas)) for s in sementes], processos)
    return [linha for lote in resultados for linha in lote]


SEMENTE_EXPERIMENTO = 40_000


def executar_experimento(config: dict, pasta_resultados, n_sementes: int | None,
                         processos: int):
    """Experimento principal: grava métricas por semente, curvas KM e parâmetros."""
    import json
    import time
    from pathlib import Path

    import pandas as pd

    from .parametros_sim import carregar_parametros

    pasta = Path(pasta_resultados)
    p = carregar_parametros(config, pasta)
    if n_sementes is None:
        piloto = pasta / "simulacao" / "piloto.json"
        n_sementes = int(config["protocolo"]["repeticoes_minimas"])
        if piloto.exists():
            n_sementes = max(n_sementes, int(json.loads(piloto.read_text(encoding="utf-8"))
                                             ["sementes_recomendadas"]))
    sementes = [SEMENTE_EXPERIMENTO + i for i in range(n_sementes)]
    inicio = time.perf_counter()
    politicas = tuple(config["simulacao"].get("politicas", POLITICAS_PROTOCOLO))
    linhas = rodar_sementes(p, sementes, processos, politicas)
    saida = pasta / "experimento"
    saida.mkdir(parents=True, exist_ok=True)
    km = [{"semente": l["semente"], "politica": l["politica"], "t": t, "S": s}
          for l in linhas for t, s in enumerate(l["_km_alta"])]
    pd.DataFrame(km).to_csv(saida / "km_alta.csv", index=False)
    df = pd.DataFrame([{k: v for k, v in l.items() if not k.startswith("_")} for l in linhas])
    df.to_csv(saida / "metricas_por_semente.csv", index=False)
    registro = p.resumo()
    registro.update({"politicas": list(politicas), "sementes": sementes, "segundos": round(time.perf_counter() - inicio, 1),
                     "processos": processos})
    (saida / "parametros.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False,
                                                      default=float), encoding="utf-8")
    return df


def meia_largura_ic95(diferencas: np.ndarray) -> float:
    """Meia largura do IC 95% (t de Student) da média das diferenças pareadas."""
    from scipy import stats
    n = len(diferencas)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * np.std(diferencas, ddof=1) / np.sqrt(n))
