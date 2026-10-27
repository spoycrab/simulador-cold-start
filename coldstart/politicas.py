"""Políticas de ranqueamento A, B, B2, C e D.

B2 (análise adicional pedida na revisão do orientador): ordena pela média
posterior Beta da contratação por exame, (a0 + H_i) / (a0 + b0 + E_i), com os
mesmos dados e a mesma priori de C, mas sem sorteio. Assim, a diferença C - B
se decompõe em (C - B2), o efeito da exploração, e (B2 - B), o efeito de
trocar o sinal (contratações por exame em vez de estrelas). B2 não sorteia
nada; incluí-la não altera os resultados das demais políticas.

Cada política recebe os candidatos de uma faixa (fornecedores ativos) e a
matriz de elegibilidade de um lote (consultas x candidatos: compatível com a
consulta e com capacidade), e devolve uma matriz (consultas x K) com os
índices dos fornecedores exibidos nas posições 1..K (-1 quando faltam
candidatos).

As políticas observam apenas o histórico: notas, número de avaliações,
exames, contratações e idade. A qualidade latente nunca é usada aqui.
Empates são desfeitos por uma chave aleatória fixa por fornecedor, igual em
todas as políticas.
"""

from __future__ import annotations

import numpy as np

from .parametros_sim import ParametrosSimulacao

DESEMPATE = 1e-9
POLITICAS = ("A", "B", "B2", "C", "D")          # todas as políticas implementadas
POLITICAS_PROTOCOLO = ("A", "B", "C", "D")      # as do protocolo pré-especificado


def escore_media(est, cand: np.ndarray, p: ParametrosSimulacao) -> np.ndarray:
    """A: média aritmética das notas; sem avaliação recebe a nota mínima."""
    n = est.n_aval[cand]
    return np.where(n > 0, est.soma_notas[cand] / np.maximum(n, 1), p.escore_sem_aval)


def escore_bayes(est, cand: np.ndarray, p: ParametrosSimulacao) -> np.ndarray:
    """B: s_i = (C m + soma das notas) / (C + n_i)."""
    return (p.C_bayes * p.media_global + est.soma_notas[cand]) / (p.C_bayes + est.n_aval[cand])


def priori_beta(taxa_global: float, p: ParametrosSimulacao) -> tuple[float, float]:
    """Priori de C e B2: centrada na taxa global de contratação por tentativa, com força kappa."""
    return p.kappa * taxa_global, p.kappa * (1.0 - taxa_global)


def escore_beta_media(est, cand: np.ndarray, taxa_global: float,
                      p: ParametrosSimulacao) -> np.ndarray:
    """B2: média posterior (a0 + H_i) / (a0 + b0 + E_i), a mesma distribuição de C, sem sorteio."""
    a0, b0 = priori_beta(taxa_global, p)
    return (a0 + est.H[cand]) / (a0 + b0 + np.maximum(est.E[cand], est.H[cand]))


def _permutacao(escore: np.ndarray, est, cand: np.ndarray) -> np.ndarray:
    """Posições dos candidatos em ordem decrescente de escore (desempate fixo)."""
    return np.argsort(-(escore + DESEMPATE * est.chave[cand]), kind="stable")


def primeiros_k(elegivel_ordenada: np.ndarray, K: int) -> np.ndarray:
    """Para cada linha, as colunas dos K primeiros elegíveis (-1 se faltarem)."""
    nq = elegivel_ordenada.shape[0]
    acumulado = np.cumsum(elegivel_ordenada, axis=1)
    selecionado = elegivel_ordenada & (acumulado <= K)
    linhas, colunas = np.nonzero(selecionado)
    saida = np.full((nq, K), -1, dtype=np.int64)
    saida[linhas, acumulado[linhas, colunas] - 1] = colunas
    return saida


def _para_fornecedores(colunas: np.ndarray, ordem: np.ndarray) -> np.ndarray:
    return np.where(colunas >= 0, ordem[np.maximum(colunas, 0)], -1)


def listas_por_escore(escore, est, cand, elegivel, K) -> np.ndarray:
    """A e B: cada consulta vê os K melhores elegíveis pelo escore."""
    perm = _permutacao(escore, est, cand)
    return _para_fornecedores(primeiros_k(elegivel[:, perm], K), cand[perm])


def listas_cota(est, cand, elegivel, t: int, p: ParametrosSimulacao) -> np.ndarray:
    """D: ordem de B com posições reservadas a novatos (idade < janela).

    Nas posições reservadas entra o melhor novato elegível ainda não listado;
    nas demais, o melhor elegível restante, novato ou não. A cota é uma
    garantia mínima: nunca rebaixa um novato que B já colocaria acima.
    """
    perm = _permutacao(escore_bayes(est, cand, p), est, cand)
    ordem = cand[perm]
    eleg = elegivel[:, perm]
    novato = (t - est.entrada[ordem]) < p.janela_novidade
    eleg_novato = eleg & novato[None, :]
    nq = eleg.shape[0]
    linhas = np.arange(nq)
    tomado = np.zeros_like(eleg)
    saida = np.full((nq, p.K), -1, dtype=np.int64)
    reservadas = set(p.posicoes_reservadas)
    for k in range(p.K):
        livre = eleg & ~tomado
        tem_livre = livre.any(axis=1)
        coluna = np.argmax(livre, axis=1)
        if (k + 1) in reservadas:
            livre_novato = eleg_novato & ~tomado
            tem_novato = livre_novato.any(axis=1)
            coluna = np.where(tem_novato, np.argmax(livre_novato, axis=1), coluna)
        saida[tem_livre, k] = coluna[tem_livre]
        tomado[linhas[tem_livre], coluna[tem_livre]] = True
    return _para_fornecedores(saida, ordem)


def listas_thompson(est, cand, elegivel, rng: np.random.Generator,
                    taxa_global: float, p: ParametrosSimulacao) -> np.ndarray:
    """C: Thompson Sampling Beta-Bernoulli, uma amostra por consulta e candidato.

    Sucesso = contratação; tentativa = exame (ou impressão, conforme o YAML).
    Priori centrada na taxa global de contratação por tentativa, com força kappa.
    """
    a0, b0 = priori_beta(taxa_global, p)
    alfa = a0 + est.H[cand]
    beta = b0 + np.maximum(est.E[cand] - est.H[cand], 0)
    nq, n = elegivel.shape
    theta = rng.beta(alfa, beta, size=(nq, n)) + DESEMPATE * est.chave[cand]
    theta[~elegivel] = -np.inf
    k = min(p.K, n)
    topo = np.argpartition(-theta, k - 1, axis=1)[:, :k]
    ordem = np.argsort(-np.take_along_axis(theta, topo, axis=1), axis=1)
    topo = np.take_along_axis(topo, ordem, axis=1)
    valido = np.isfinite(np.take_along_axis(theta, topo, axis=1))
    saida = np.full((nq, p.K), -1, dtype=np.int64)
    saida[:, :k] = np.where(valido, cand[topo], -1)
    return saida


def ranquear(politica: str, est, cand: np.ndarray, elegivel: np.ndarray, t: int,
             p: ParametrosSimulacao, gerador_ts, taxa_global: float) -> np.ndarray:
    if politica == "A":
        return listas_por_escore(escore_media(est, cand, p), est, cand, elegivel, p.K)
    if politica == "B":
        return listas_por_escore(escore_bayes(est, cand, p), est, cand, elegivel, p.K)
    if politica == "B2":
        return listas_por_escore(escore_beta_media(est, cand, taxa_global, p), est, cand,
                                 elegivel, p.K)
    if politica == "C":
        return listas_thompson(est, cand, elegivel, gerador_ts(), taxa_global, p)
    if politica == "D":
        return listas_cota(est, cand, elegivel, t, p)
    raise ValueError(f"Política desconhecida: {politica}")
