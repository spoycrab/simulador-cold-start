"""Calibração do simulador, validação e piloto de precisão.

Calibração (sob a política de referência, que representa a prática atual;
padrão D, reputação com impulso a novatos; configurável no YAML):

  parâmetro                         alvo (etapa 1)
  clientes por rodada (demanda)  -> avaliações por fornecedor por mês
  compatibilidade f              -> fração da coorte de 12 a 23 meses sem avaliação

Procedimento:
  1. para cada f da grade, a demanda é ajustada (regula falsi em escala log)
     para reproduzir o volume de avaliações;
  2. f é escolhido por interpolação (escala log) da fração sem avaliação;
  3. a demanda é reajustada no f escolhido, e f é reinterpolado com os pontos
     mais próximos até a fração sem avaliação ficar dentro da tolerância.

A tabela de identificação (um f por linha) mostra como cada alvo responde a
cada parâmetro. u0 não entra: com esses dois alvos, ele não se separa da
demanda (ambos só mudam o volume), e por isso é tratado como hipótese.

Validação: com os parâmetros calibrados, compara momentos simulados e
observados, incluindo alvos que não entraram no ajuste.

Piloto: roda o desenho completo em poucas sementes e informa apenas a
precisão Monte Carlo das diferenças primárias e o tempo de execução. As
estimativas de efeito não são exibidas, para preservar o pré-registro.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .experimento import mapear, meia_largura_ic95, momentos_referencia, rodar_sementes
from .parametros_sim import ParametrosSimulacao, carregar_parametros
from .saidas import salvar_tabela

ALVO_VOLUME = "avaliacoes_por_fornecedor_mes"
ALVO_SEM_AVAL = "prop_sem_avaliacao_coorte_12_23m"   # padrão; configurável no YAML
SEMENTE_CALIBRACAO = 10_000   # sementes de calibração e validação não se repetem
SEMENTE_VALIDACAO = 20_000
SEMENTE_PILOTO = 30_000


def _momentos_medios(p: ParametrosSimulacao, sementes, processos) -> dict:
    lista = mapear(momentos_referencia, [(p, s) for s in sementes], processos)
    return {k: float(np.mean([m[k] for m in lista])) for k in lista[0]}


def ajustar_demanda(p: ParametrosSimulacao, alvo: float, sementes, processos,
                    tolerancia: float, max_iter: int, log: list,
                    chave_f: str = ALVO_SEM_AVAL) -> tuple[float, dict]:
    """Encontra clientes por rodada que reproduzem o volume de avaliações."""
    contratacoes_alvo = alvo / p.p_review * p.N0
    palpite = contratacoes_alvo / 0.3
    cache = {}

    def avaliar(nc: float) -> float:
        nc = float(round(nc))
        if nc not in cache:
            m = _momentos_medios(p.com(clientes_por_rodada=nc), sementes, processos)
            cache[nc] = m
            log.append({"compatibilidade": p.compatibilidade, "clientes_por_rodada": nc,
                        ALVO_VOLUME: m[ALVO_VOLUME], chave_f: m[chave_f]})
        return cache[nc][ALVO_VOLUME] - alvo

    lo, hi = math.log(palpite / 3), math.log(palpite * 3)
    f_lo, f_hi = avaliar(math.exp(lo)), avaliar(math.exp(hi))
    for _ in range(4):
        if f_lo <= 0 <= f_hi:
            break
        if f_lo > 0:
            lo, f_lo = lo - math.log(3), avaliar(math.exp(lo - math.log(3)))
        if f_hi < 0:
            hi, f_hi = hi + math.log(3), avaliar(math.exp(hi + math.log(3)))
    if not f_lo <= 0 <= f_hi:
        raise RuntimeError("Não foi possível cercar a demanda que reproduz o volume de avaliações.")

    lado = 0
    melhor = min(cache, key=lambda nc: abs(cache[nc][ALVO_VOLUME] - alvo))
    for _ in range(max_iter):
        meio = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)   # regula falsi (Illinois)
        f_meio = avaliar(math.exp(meio))
        melhor = min(cache, key=lambda nc: abs(cache[nc][ALVO_VOLUME] - alvo))
        if abs(cache[melhor][ALVO_VOLUME] - alvo) / alvo <= tolerancia:
            break
        if f_meio < 0:
            lo, f_lo = meio, f_meio
            if lado == -1:
                f_hi /= 2
            lado = -1
        else:
            hi, f_hi = meio, f_meio
            if lado == 1:
                f_lo /= 2
            lado = 1
    return melhor, cache[melhor]


def calibrar_simulador(config: dict, pasta_resultados: str | Path, processos: int,
                       rapido: bool = False) -> dict:
    pasta_resultados = Path(pasta_resultados)
    saida = pasta_resultados / "simulacao"
    saida.mkdir(parents=True, exist_ok=True)
    cfg = config["simulacao"]["calibracao"]
    cal = json.loads((pasta_resultados / "calibracao" / "calibracao.json").read_text(encoding="utf-8"))
    alvos = {**cal["alvos_validacao"], **cal["alvos_calibracao"]}
    chave_f = cfg.get("alvo_compatibilidade", ALVO_SEM_AVAL)
    if chave_f not in alvos:
        raise FileNotFoundError(f"Alvo '{chave_f}' ausente em calibracao.json. "
                                "Rode de novo: python -m coldstart calibrar")
    alvo_volume = alvos[ALVO_VOLUME]["valor"]
    alvo_sem = alvos[chave_f]["valor"]
    base = carregar_parametros(config, pasta_resultados, exigir_calibracao=False)

    n_sementes = 1 if rapido else int(cfg["sementes"])
    sementes = [SEMENTE_CALIBRACAO + i for i in range(n_sementes)]
    grade = cfg["grade_compatibilidade"][::2] if rapido else cfg["grade_compatibilidade"]
    tol = 0.05 if rapido else float(cfg["tolerancia_relativa"])
    max_iter = 6 if rapido else int(cfg["max_iteracoes"])
    log: list[dict] = []
    inicio = time.perf_counter()

    curva = []
    for f in grade:
        nc, m = ajustar_demanda(base.com(compatibilidade=f), alvo_volume, sementes,
                                processos, tol, max_iter, log, chave_f)
        curva.append({"compatibilidade": f, "clientes_por_rodada": nc,
                      ALVO_VOLUME: m[ALVO_VOLUME], chave_f: m[chave_f]})
        print(f"  f = {f:<5} demanda = {nc:>8.0f}  volume = {m[ALVO_VOLUME]:.3f}  "
              f"sem avaliação = {m[chave_f]:.3f}")

    # refinamento: reinterpola f com os pontos mais próximos até atingir a tolerância
    pontos = list(curva)
    tol_sem = float(cfg.get("tolerancia_sem_avaliacao", 0.01))
    for iteracao in range(1 if rapido else int(cfg.get("max_refinamentos", 4))):
        f_escolhido = _interpolar_f(pontos, alvo_sem, chave_f)
        nc, m = ajustar_demanda(base.com(compatibilidade=f_escolhido), alvo_volume, sementes,
                                processos, tol, max_iter, log, chave_f)
        escolhido = {"compatibilidade": f_escolhido, "clientes_por_rodada": nc,
                     ALVO_VOLUME: m[ALVO_VOLUME], chave_f: m[chave_f]}
        pontos.append(escolhido)
        print(f"  refinamento {iteracao + 1}: f = {f_escolhido:.4f}  demanda = {nc:.0f}  "
              f"sem avaliação = {m[chave_f]:.3f}")
        if abs(m[chave_f] - alvo_sem) <= tol_sem:
            break
    resultado = {
        "escolhido": escolhido,
        "alvos": {ALVO_VOLUME: alvo_volume, chave_f: alvo_sem},
        "alvo_compatibilidade": chave_f,
        "curva_identificacao": curva,
        "avaliacoes": log,
        "sementes": sementes, "rapido": rapido,
        "politica_referencia": base.politica_referencia,
        "hipoteses_fixas": {"u0": base.u0, "tau": base.tau, "gamma": base.gamma,
                            "p_review": base.p_review, "churn": base.churn,
                            "capacidade": base.capacidade},
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    (saida / "calibracao_simulador.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    tabela = pd.DataFrame(curva).rename(columns={
        "compatibilidade": "Compatibilidade f", "clientes_por_rodada": "Clientes por rodada",
        ALVO_VOLUME: "Avaliações/mês (sim.)", chave_f: "Sem avaliação, 12 a 23 meses (sim.)"})
    salvar_tabela(tabela, saida, "identificacao",
                  "Identificação: demanda ajustada ao volume para cada compatibilidade",
                  "tab:identificacao", casas=3)
    return resultado


def _interpolar_f(curva: list[dict], alvo: float, chave: str = ALVO_SEM_AVAL) -> float:
    """Interpola f (em escala log) entre os pontos vizinhos que cercam o alvo."""
    pontos = sorted(curva, key=lambda c: c["compatibilidade"])
    pares = [(a, b) for a, b in zip(pontos[:-1], pontos[1:])
             if (a[chave] - alvo) * (b[chave] - alvo) <= 0]
    for a, b in pares[:1]:
        ya, yb = a[chave], b[chave]
        if (ya - alvo) * (yb - alvo) <= 0 and ya != yb:
            la, lb = math.log(a["compatibilidade"]), math.log(b["compatibilidade"])
            return float(math.exp(la + (alvo - ya) * (lb - la) / (yb - ya)))
    mais_proximo = min(pontos, key=lambda c: abs(c[chave] - alvo))
    print("  AVISO: a fração sem avaliação não cruza o alvo na grade; "
          f"usado o f mais próximo ({mais_proximo['compatibilidade']}).")
    return float(mais_proximo["compatibilidade"])


def validar(config: dict, pasta_resultados: str | Path, processos: int) -> pd.DataFrame:
    """Compara momentos simulados (parâmetros calibrados) com os observados."""
    pasta_resultados = Path(pasta_resultados)
    cal = json.loads((pasta_resultados / "calibracao" / "calibracao.json").read_text(encoding="utf-8"))
    p = carregar_parametros(config, pasta_resultados)
    n = int(config["simulacao"]["validacao"]["sementes"])
    lista = mapear(momentos_referencia, [(p, SEMENTE_VALIDACAO + i) for i in range(n)], processos)
    usados = {ALVO_VOLUME, config["simulacao"]["calibracao"].get("alvo_compatibilidade", ALVO_SEM_AVAL)}
    linhas = []
    for nome, item in {**cal["alvos_calibracao"], **cal["alvos_validacao"]}.items():
        if nome not in lista[0]:
            continue
        if nome in usados:
            papel = "calibração"
        elif item.get("comparavel", True):
            papel = "validação"
        else:
            papel = "referência (não comparável)"
        valores = np.array([m[nome] for m in lista])
        media = float(valores.mean())
        hw = meia_largura_ic95(valores)
        linhas.append({"Momento": nome, "Papel": papel, "Dados": item["valor"],
                       "Simulado": media, "IC95 inf.": media - hw, "IC95 sup.": media + hw,
                       "Diferença relativa (%)": 100 * (media - item["valor"]) / item["valor"]})
    ordem = {"calibração": 0, "validação": 1, "referência (não comparável)": 2}
    linhas.sort(key=lambda l: ordem[l["Papel"]])
    tabela = pd.DataFrame(linhas)
    salvar_tabela(tabela, pasta_resultados / "simulacao", "validacao",
                  f"Momentos observados e simulados (política de referência {p.politica_referencia}, "
                  "parâmetros calibrados)",
                  "tab:validacao", casas=3)
    return tabela


def piloto(config: dict, pasta_resultados: str | Path, processos: int) -> dict:
    """Precisão Monte Carlo das comparações primárias, sem exibir estimativas de efeito."""
    pasta_resultados = Path(pasta_resultados)
    cfg = config["simulacao"]["piloto"]
    p = carregar_parametros(config, pasta_resultados)
    n = int(cfg["sementes"])
    inicio = time.perf_counter()
    linhas = rodar_sementes(p, [SEMENTE_PILOTO + i for i in range(n)], processos)
    segundos = time.perf_counter() - inicio
    df = pd.DataFrame(linhas).set_index(["semente", "politica"])

    limites = {"ndcg10": float(cfg["meia_largura_max_ndcg"]),
               "rmst_alta": float(cfg["meia_largura_max_rmst"])}
    precisao = []
    for metrica, limite in limites.items():
        for a, b in config["protocolo"]["comparacoes_primarias"]:
            dif = (df.xs(a, level="politica")[metrica] - df.xs(b, level="politica")[metrica]).to_numpy()
            hw = meia_largura_ic95(dif)
            necessarias = int(math.ceil(n * (hw / limite) ** 2)) if hw > 0 else n
            precisao.append({"metrica": metrica, "comparacao": f"{a} vs {b}",
                             "meia_largura_ic95": hw, "limite": limite,
                             "sementes_necessarias": max(necessarias,
                                                         int(config["protocolo"]["repeticoes_minimas"]))})
    recomendadas = max(item["sementes_necessarias"] for item in precisao)
    resultado = {
        "sementes_piloto": n, "segundos_total": round(segundos, 1),
        "segundos_por_semente_em_um_nucleo": round(segundos * min(processos, n) / n, 1),
        "processos": processos, "politica_referencia": p.politica_referencia,
        "precisao": precisao, "sementes_recomendadas": recomendadas,
        "estimativa_minutos_experimento": round(recomendadas * segundos / n / 60, 1),
        "observacao": "Estimativas de efeito omitidas para preservar o pré-registro.",
        "n_coorte_alta_medio": float(df["n_coorte_alta"].mean()),
        "consultas_sem_oferta_medio": float(df["consultas_sem_oferta"].mean()),
    }
    saida = pasta_resultados / "simulacao"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "piloto.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return resultado
