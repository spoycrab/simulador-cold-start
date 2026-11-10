"""Métricas de uma execução e momentos simulados para calibração e validação."""

from __future__ import annotations

import numpy as np

from .motor import SEM_EVENTO, Estado
from .parametros_sim import ParametrosSimulacao

# ---------------------------------------------------------------------------
# Sobrevivência em tempo discreto
# ---------------------------------------------------------------------------


def kaplan_meier(tempos: np.ndarray, eventos: np.ndarray, h: int) -> np.ndarray:
    """Curva S(t) = P(T > t) para t = 0..h, com censura à direita.

    `tempos` em rodadas (1 = contratado na rodada de entrada); para censurados,
    é o número de rodadas observadas.
    """
    S = np.ones(h + 1)
    for j in range(1, h + 1):
        em_risco = np.sum(tempos >= j)
        ocorridos = np.sum((tempos == j) & eventos)
        S[j] = S[j - 1] * (1 - ocorridos / em_risco) if em_risco > 0 else S[j - 1]
    return S


def rmst(curva: np.ndarray, h: int) -> float:
    """Média restrita do tempo até o evento: E[min(T, h)] = soma de S(t), t = 0..h-1."""
    return float(curva[:h].sum())


def incidencia_cumulativa(tempos: np.ndarray, causa: np.ndarray, h: int) -> dict:
    """Aalen–Johansen em tempo discreto, com dois eventos competitivos.

    `causa`: 1 = contratação, 2 = saída do mercado, 0 = censura administrativa.
    `tempos`: rodada do evento (1 = rodada de entrada) ou rodadas observadas.
    A saída ocorre no início da rodada, antes das consultas; por isso, numa
    mesma rodada, as saídas são retiradas do conjunto em risco antes das
    contratações. Devolve F_contratacao(t) e F_saida(t) para t = 0..h.
    """
    F1, F2 = np.zeros(h + 1), np.zeros(h + 1)
    S = 1.0
    for j in range(1, h + 1):
        em_risco = np.sum(tempos >= j)
        saidas = np.sum((tempos == j) & (causa == 2))
        contratos = np.sum((tempos == j) & (causa == 1))
        F1[j], F2[j] = F1[j - 1], F2[j - 1]
        if em_risco == 0:
            continue
        F2[j] += S * saidas / em_risco
        S *= 1 - saidas / em_risco
        restantes = em_risco - saidas
        if restantes > 0:
            F1[j] += S * contratos / restantes
            S *= 1 - contratos / restantes
    return {"contratacao": F1, "saida": F2}


def gini(valores: np.ndarray) -> float:
    x = np.sort(np.asarray(valores, dtype=float))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return float("nan")
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Coorte de entrantes (tempo até a primeira contratação)
# ---------------------------------------------------------------------------


def coorte_ttfc(est: Estado, p: ParametrosSimulacao, horizonte: int) -> dict:
    """Entrantes da fase de avaliação observáveis por `horizonte` rodadas.

    Evento: primeira contratação. Censura: saída do mercado ou fim do horizonte
    (estimando contrafactual: tempo até contratar se o fornecedor permanecesse).
    `tempo_cr` e `causa` servem à análise complementar de riscos competitivos,
    em que a saída antes da contratação é um evento, na rodada da saída.
    """
    TT = p.T_total
    entrada = est.entrada
    coorte = np.flatnonzero((entrada >= p.aquecimento) & (entrada <= TT - horizonte))
    e = entrada[coorte]
    contr = est.primeira_contratacao[coorte]
    saida = est.saida[coorte]
    contratado = contr != SEM_EVENTO
    t_evento = np.where(contratado, contr - e + 1, np.iinfo(np.int64).max)
    t_censura = np.where(saida != SEM_EVENTO, saida - e, horizonte)
    t_censura = np.minimum(t_censura, horizonte)
    evento = contratado & (t_evento <= t_censura)
    tempo = np.where(evento, t_evento, t_censura)
    saiu = ~evento & (saida != SEM_EVENTO) & (saida - e + 1 <= horizonte)
    tempo_cr = np.where(saiu, saida - e + 1, tempo)
    causa = np.where(evento, 1, np.where(saiu, 2, 0))
    return {"tempo": tempo, "evento": evento, "alta": est.q[coorte] >= p.limiar_alta,
            "indices": coorte, "tempo_cr": tempo_cr, "causa": causa}


def metricas_execucao(est: Estado, p: ParametrosSimulacao) -> dict:
    """Métricas primárias e secundárias de uma política em uma semente."""
    acc = est.acc
    atendidas = acc.consultas - acc.consultas_sem_oferta
    h = p.horizonte_h
    ttfc = coorte_ttfc(est, p, h)
    alta = ttfc["alta"]
    km_alta = kaplan_meier(ttfc["tempo"][alta], ttfc["evento"][alta], h)
    km_todos = kaplan_meier(ttfc["tempo"], ttfc["evento"], h)

    cif_alta = incidencia_cumulativa(ttfc["tempo_cr"][alta], ttfc["causa"][alta], h)
    cif_todos = incidencia_cumulativa(ttfc["tempo_cr"], ttfc["causa"], h)

    janela = coorte_ttfc(est, p, p.janela_descoberta)
    descoberta = janela["evento"][janela["alta"]]

    elegiveis = est.consultas_elegiveis > 0
    expo_norm = est.exposicao[elegiveis] / est.consultas_elegiveis[elegiveis]

    return {
        "ndcg10": acc.soma_ndcg / atendidas if atendidas else float("nan"),
        "precisao10": acc.soma_precisao / atendidas if atendidas else float("nan"),
        "rmst_alta": rmst(km_alta, h),
        "rmst_todos": rmst(km_todos, h),
        "p_contratacao_ate_h_alta": float(1 - km_alta[h]),
        "cif_contratacao_h_alta": float(cif_alta["contratacao"][h]),
        "cif_saida_sem_contrato_h_alta": float(cif_alta["saida"][h]),
        "cif_contratacao_h_todos": float(cif_todos["contratacao"][h]),
        "descoberta_alta": float(descoberta.mean()) if len(descoberta) else float("nan"),
        "gini_exposicao": gini(expo_norm),
        "arrependimento_por_contratacao": (acc.soma_arrependimento / acc.contratacoes
                                           if acc.contratacoes else float("nan")),
        "conversao": acc.contratacoes / acc.consultas if acc.consultas else float("nan"),
        "consultas_sem_oferta": acc.consultas_sem_oferta / acc.consultas if acc.consultas else 0.0,
        "elegiveis_por_consulta": acc.soma_elegiveis / acc.consultas if acc.consultas else 0.0,
        "n_coorte": int(len(ttfc["tempo"])),
        "n_coorte_alta": int(alta.sum()),
        "fornecedores_ativos_fim": int(est.ativo.sum()),
    }


def curva_km_alta(est: Estado, p: ParametrosSimulacao) -> np.ndarray:
    ttfc = coorte_ttfc(est, p, p.horizonte_h)
    return kaplan_meier(ttfc["tempo"][ttfc["alta"]], ttfc["evento"][ttfc["alta"]], p.horizonte_h)


def curva_cif_alta(est: Estado, p: ParametrosSimulacao) -> np.ndarray:
    """Incidência cumulativa da contratação antes da saída, novatos de alta qualidade."""
    ttfc = coorte_ttfc(est, p, p.horizonte_h)
    alta = ttfc["alta"]
    return incidencia_cumulativa(ttfc["tempo_cr"][alta], ttfc["causa"][alta],
                                 p.horizonte_h)["contratacao"]


# ---------------------------------------------------------------------------
# Momentos simulados (mesmas definições da etapa 1)
# ---------------------------------------------------------------------------


def momentos_simulados(est: Estado, p: ParametrosSimulacao, janela: int = 24) -> dict:
    """Momentos comparáveis aos alvos de calibração e de validação.

    Como nos dados, só entram fornecedores ativos ao fim (sobreviventes), e a
    idade é o número de rodadas desde a entrada.
    """
    TT = p.T_total
    entrada, saida = est.entrada, est.saida
    entrou = (entrada >= 0) & (entrada < TT)
    idade_final = TT - entrada

    # alvo de calibração 1: sem avaliação nos primeiros `janela` meses (sobreviventes)
    coorte = entrou & (idade_final >= janela) & (saida == SEM_EVENTO)
    prim = est.primeira_avaliacao
    avaliado_janela = (prim != SEM_EVENTO) & (prim - entrada < janela)
    sem_janela = ~avaliado_janela[coorte]

    # alvo de calibração 2: avaliações por mês nos últimos 12 meses (12 meses ou mais)
    maduros = est.ativo & (idade_final >= 12)
    aval_12m = est.aval_por_rodada[maduros, TT - 12:].sum(axis=1) / 12

    # validação
    meses_1a = (prim - entrada + 0.5)[coorte & avaliado_janela]
    jovem = est.ativo & entrou & (idade_final >= 12) & (idade_final < 24)
    meses_1a_jovem = (prim - entrada + 0.5)[jovem & (prim != SEM_EVENTO)]
    return {
        "prop_sem_avaliacao_janela": float(sem_janela.mean()) if sem_janela.size else float("nan"),
        "avaliacoes_por_fornecedor_mes": float(aval_12m.mean()) if aval_12m.size else float("nan"),
        "mediana_meses_ate_1a_avaliacao": float(np.median(meses_1a)) if meses_1a.size else float("nan"),
        "prop_sem_avaliacao_coorte_12_23m": float((est.n_aval[jovem] == 0).mean()) if jovem.any() else float("nan"),
        "mediana_avaliacoes_coorte_12_23m": float(np.median(est.n_aval[jovem])) if jovem.any() else float("nan"),
        "prop_1a_avaliacao_ate_3m_coorte_12_23m": (float((meses_1a_jovem <= 3).sum() / jovem.sum())
                                                   if jovem.any() else float("nan")),
        "mediana_meses_ate_1a_avaliacao_coorte_12_23m": (float(np.median(meses_1a_jovem))
                                                         if meses_1a_jovem.size else float("nan")),
        "n_coorte_janela": int(coorte.sum()),
        "n_coorte_12_23m": int(jovem.sum()),
    }
