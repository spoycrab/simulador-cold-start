"""Estado do mercado e motor de rodadas.

Sequência de cada rodada t (um mês):
  1. saídas exógenas (probabilidade churn por fornecedor ativo);
  2. entradas exógenas, sem histórico;
  3. a capacidade mensal de cada fornecedor é restabelecida;
  4. as consultas de cada faixa chegam em lotes; cada consulta é compatível
     com cada fornecedor ativo da faixa com probabilidade f (datas, local e
     requisitos; sorteio exógeno); a política ordena os elegíveis
     (compatíveis e com capacidade) e exibe K;
  5. a posição k é examinada com probabilidade min(1, gamma / log2(1 + k));
  6. o cliente escolhe pelo modelo logit entre os examinados e a opção de
     não contratar (utilidade u0); quem chega a um fornecedor sem capacidade
     no mesmo lote não é atendido;
  7. exames e contratações atualizam o aprendizado da política C;
  8. cada contratação gera uma avaliação com probabilidade p_review, visível
     ao fim da rodada ou da rodada seguinte.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .fluxos import Fluxos
from .parametros_sim import ParametrosSimulacao
from .politicas import ranquear

SEM_EVENTO = -1
NAO_ENTROU = np.iinfo(np.int64).max // 2


@dataclass
class Acumuladores:
    """Somatórios das métricas do cliente, coletados só na fase de avaliação."""
    consultas: int = 0
    consultas_sem_oferta: int = 0
    contratacoes: int = 0
    nao_atendidas: int = 0
    soma_ndcg: float = 0.0
    soma_precisao: float = 0.0
    soma_arrependimento: float = 0.0


@dataclass
class Estado:
    ativo: np.ndarray
    faixa: np.ndarray
    q: np.ndarray
    chave: np.ndarray
    entrada: np.ndarray
    saida: np.ndarray
    n_aval: np.ndarray
    soma_notas: np.ndarray
    E: np.ndarray
    H: np.ndarray
    primeira_contratacao: np.ndarray
    primeira_avaliacao: np.ndarray
    contador_contratacoes: np.ndarray
    aval_por_rodada: np.ndarray
    exposicao: np.ndarray
    consultas_elegiveis: np.ndarray
    pendentes: tuple = field(default_factory=lambda: (np.empty(0, np.int64), np.empty(0)))
    acc: Acumuladores = field(default_factory=Acumuladores)

    def copia(self) -> "Estado":
        return copy.deepcopy(self)

    def iniciar_coleta(self) -> None:
        """Zera os acumuladores da fase de avaliação (após o aquecimento)."""
        self.exposicao[:] = 0.0
        self.consultas_elegiveis[:] = 0
        self.acc = Acumuladores()


def inicializar(p: ParametrosSimulacao, fl: Fluxos) -> Estado:
    """Estado inicial: incumbentes da amostra com histórico; entrantes reservados."""
    N, N0 = fl.N_max, p.N0
    q = p.qualidade_de_uniforme(fl.u_qualidade)
    q[:N0] = np.where(np.isnan(p.q0), q[:N0], p.q0)
    faixa = np.empty(N, dtype=np.int64)
    faixa[:N0] = p.faixa0
    faixa[N0:] = fl.faixa_entrante
    entrada = np.full(N, NAO_ENTROU, dtype=np.int64)
    entrada[:N0] = -p.idade0
    ativo = np.zeros(N, dtype=bool)
    ativo[:N0] = True
    n_aval = np.zeros(N, dtype=np.int64)
    n_aval[:N0] = p.n_aval0
    soma = np.zeros(N)
    soma[:N0] = p.soma0
    return Estado(
        ativo=ativo, faixa=faixa, q=q, chave=fl.chave.copy(), entrada=entrada,
        saida=np.full(N, SEM_EVENTO, dtype=np.int64), n_aval=n_aval, soma_notas=soma,
        E=np.zeros(N, dtype=np.int64), H=np.zeros(N, dtype=np.int64),
        primeira_contratacao=np.full(N, SEM_EVENTO, dtype=np.int64),
        primeira_avaliacao=np.full(N, SEM_EVENTO, dtype=np.int64),
        contador_contratacoes=np.zeros(N, dtype=np.int64),
        aval_por_rodada=np.zeros((N, p.T_total), dtype=np.int16),
        exposicao=np.zeros(N), consultas_elegiveis=np.zeros(N, dtype=np.int64),
    )


def ordem_de_ocorrencia(valores: np.ndarray) -> np.ndarray:
    """Para cada elemento, quantas vezes o mesmo valor apareceu antes (0, 1, 2, ...)."""
    if len(valores) == 0:
        return np.empty(0, dtype=np.int64)
    ordem = np.argsort(valores, kind="stable")
    ordenados = valores[ordem]
    inicios = np.r_[0, np.flatnonzero(ordenados[1:] != ordenados[:-1]) + 1]
    tamanhos = np.diff(np.r_[inicios, len(ordenados)])
    posicao = np.arange(len(ordenados)) - np.repeat(inicios, tamanhos)
    resultado = np.empty(len(valores), dtype=np.int64)
    resultado[ordem] = posicao
    return resultado


def _registrar_avaliacoes(est: Estado, prov: np.ndarray, notas: np.ndarray, t: int) -> None:
    if len(prov) == 0:
        return
    np.add.at(est.n_aval, prov, 1)
    np.add.at(est.soma_notas, prov, notas)
    np.add.at(est.aval_por_rodada[:, t], prov, 1)
    primeiras = prov[est.primeira_avaliacao[prov] == SEM_EVENTO]
    est.primeira_avaliacao[primeiras] = t


def rodada(est: Estado, t: int, politica: str, p: ParametrosSimulacao, fl: Fluxos,
           coletar: bool) -> None:
    """Executa a rodada t sob a política informada."""
    # 1-2. saídas e entradas exógenas
    saem = est.ativo & (fl.u_saida[:, t] < p.churn)
    est.ativo[saem] = False
    est.saida[saem] = t
    i0, n_ent = fl.inicio_entrada[t], fl.n_entradas[t]
    est.ativo[i0:i0 + n_ent] = True
    est.entrada[i0:i0 + n_ent] = t

    # 3. capacidade e taxa global (priori da política C)
    capacidade = np.where(est.ativo, p.capacidade, 0).astype(np.int64)
    taxa_global = float(est.H.sum() / est.E.sum()) if est.E.sum() > 0 else 0.5
    taxa_global = min(max(taxa_global, 1e-3), 1 - 1e-3)
    p_exame, v, K = p.p_exame, p.v, p.K
    lote = p.consultas_por_lote
    novas_prov, novas_notas, adiadas_prov, adiadas_notas = [], [], [], []

    for g in range(p.n_faixas):
        total = int(fl.consultas[t, g])
        cand = np.flatnonzero(est.ativo & (est.faixa == g))   # igual em todas as políticas
        for b, inicio in enumerate(range(0, total, lote)):
            nq = min(lote, total - inicio)
            sorteio = fl.gerador_lote(t, g, b)
            u_exame = sorteio.random((nq, K))
            u_escolha = sorteio.random(nq)
            u_compat = sorteio.random((nq, len(cand)))
            if coletar:
                est.acc.consultas += nq
            if len(cand) == 0:
                if coletar:
                    est.acc.consultas_sem_oferta += nq
                continue
            # elegível = compatível com a consulta (exógeno) e com capacidade
            elegivel = (u_compat < p.compatibilidade) & (capacidade[cand] > 0)[None, :]

            # 4. ranqueamento
            listas = ranquear(politica, est, cand, elegivel, t, p,
                              lambda: fl.gerador_thompson(t, g, b), taxa_global)
            valida = listas >= 0
            idx = np.where(valida, listas, 0)

            # 5-6. exame e escolha logit com opção externa
            examinado = valida & (u_exame < p_exame)
            q_lista = np.where(valida, est.q[idx], 0.0)
            util = np.where(examinado, p.tau * q_lista, -np.inf)
            maximo = np.maximum(util.max(axis=1), p.u0)
            pesos = np.exp(util - maximo[:, None])
            peso_externo = np.exp(p.u0 - maximo)
            acumulada = np.cumsum(pesos, axis=1) / (pesos.sum(axis=1) + peso_externo)[:, None]
            contratou = u_escolha < acumulada[:, -1]
            posicao = np.argmax(u_escolha[:, None] < acumulada, axis=1)
            consultas_c = np.flatnonzero(contratou)
            escolhidos = idx[consultas_c, posicao[consultas_c]]

            # capacidade dentro do lote: atendidos na ordem de chegada
            aceito = ordem_de_ocorrencia(escolhidos) < capacidade[escolhidos]
            contratados = escolhidos[aceito]
            np.subtract.at(capacidade, contratados, 1)

            # 7. aprendizado (política C usa exames ou impressões)
            tentativa = examinado if p.tentativa_C == "exames" else valida
            np.add.at(est.E, idx[tentativa], 1)
            np.add.at(est.H, contratados, 1)
            primeiras = contratados[est.primeira_contratacao[contratados] == SEM_EVENTO]
            est.primeira_contratacao[primeiras] = t

            # 8. avaliações (sorteios da n-ésima contratação de cada fornecedor)
            n = est.contador_contratacoes[contratados] + ordem_de_ocorrencia(contratados)
            np.add.at(est.contador_contratacoes, contratados, 1)
            avalia = fl.u_avaliacao[contratados, n] < p.p_review
            notas = p.nota_de_qualidade(est.q[contratados] + p.sigma * fl.z_avaliacao[contratados, n])
            adia = fl.u_atraso[contratados, n] < p.p_atraso
            novas_prov.append(contratados[avalia & ~adia])
            novas_notas.append(notas[avalia & ~adia])
            adiadas_prov.append(contratados[avalia & adia])
            adiadas_notas.append(notas[avalia & adia])

            if coletar:
                _coletar(est, p, cand, elegivel, listas, valida, idx, q_lista,
                         consultas_c[aceito], contratados, int((~aceito).sum()))

    # avaliações visíveis ao fim da rodada: as adiadas da rodada anterior e as imediatas
    pend_prov, pend_notas = est.pendentes
    _registrar_avaliacoes(est, pend_prov, pend_notas, t)
    if novas_prov:
        _registrar_avaliacoes(est, np.concatenate(novas_prov), np.concatenate(novas_notas), t)
        est.pendentes = (np.concatenate(adiadas_prov), np.concatenate(adiadas_notas))
    else:
        est.pendentes = (np.empty(0, np.int64), np.empty(0))


def _coletar(est, p, cand, elegivel, listas, valida, idx, q_lista,
             consultas_contratadas, contratados, nao_atendidas):
    """Métricas do cliente e exposição dos fornecedores (fase de avaliação).

    O NDCG@K de cada consulta usa como ideal os K elegíveis de maior
    qualidade daquela consulta; consultas sem nenhum elegível ficam fora.
    """
    acc, v, K = est.acc, p.v, p.K
    pesos = np.broadcast_to(v, listas.shape)
    np.add.at(est.exposicao, idx[valida], pesos[valida])
    est.consultas_elegiveis[cand] += elegivel.sum(axis=0)

    com_oferta = elegivel.any(axis=1)
    acc.consultas_sem_oferta += int((~com_oferta).sum())
    q_eleg = np.where(elegivel, est.q[cand][None, :], -np.inf)
    k = min(K, q_eleg.shape[1])
    melhores = -np.sort(-np.partition(q_eleg, q_eleg.shape[1] - k, axis=1)[:, -k:], axis=1)
    melhores = np.where(np.isfinite(melhores), melhores, 0.0)
    idcg = (melhores * v[:k]).sum(axis=1)
    dcg = (q_lista * v).sum(axis=1)
    ok = com_oferta & (idcg > 0)
    acc.soma_ndcg += float((dcg[ok] / idcg[ok]).sum())
    acc.soma_precisao += float(((valida & (q_lista >= p.limiar_alta)).sum(axis=1)[ok] / K).sum())
    melhor = q_eleg.max(axis=1)
    acc.soma_arrependimento += float((melhor[consultas_contratadas] - est.q[contratados]).sum())
    acc.contratacoes += len(contratados)
    acc.nao_atendidas += nao_atendidas


def simular(p: ParametrosSimulacao, fl: Fluxos, politica: str = "B",
            est: Estado | None = None, inicio: int = 0, fim: int | None = None,
            coletar: bool = False) -> Estado:
    """Executa as rodadas [inicio, fim) sob uma política."""
    est = inicializar(p, fl) if est is None else est
    for t in range(inicio, p.T_total if fim is None else fim):
        rodada(est, t, politica, p, fl, coletar)
    return est
