"""Testes do simulador com um mercado minúsculo (executam em poucos segundos)."""

import numpy as np
import pytest

from coldstart import metricas, motor
from coldstart.experimento import rodar_semente
from coldstart.fluxos import Fluxos
from coldstart.motor import ordem_de_ocorrencia
from coldstart.parametros_sim import ParametrosSimulacao
from coldstart.politicas import listas_cota, primeiros_k


def parametros(**alteracoes) -> ParametrosSimulacao:
    rng = np.random.default_rng(0)
    n0 = 60
    avaliado = rng.random(n0) < 0.7
    n_aval = np.where(avaliado, rng.integers(1, 30, n0), 0)
    notas = rng.uniform(4.0, 5.0, n0)
    base = ParametrosSimulacao(
        faixa0=rng.integers(0, 2, n0), q0=np.where(avaliado, rng.random(n0), np.nan),
        n_aval0=n_aval, soma0=notas * n_aval, idade0=rng.integers(0, 60, n0),
        n_faixas=2, participacao_oferta=np.array([0.5, 0.5]),
        participacao_demanda=np.array([0.5, 0.5]), media_global=4.8,
        quantis_nota=np.linspace(4.0, 5.0, 101), beta_a=5.0, beta_b=1.0,
        taxa_entrada=0.05, capacidade=2, aquecimento=3, T=8, K=5,
        clientes_por_rodada=40, compatibilidade=0.5, consultas_por_lote=10,
        posicoes_reservadas=(2, 4), rho=0.4, janela_novidade=3, horizonte_h=4,
        janela_descoberta=4,
    )
    return base.com(**alteracoes)


def test_probabilidade_de_exame():
    p = parametros(gamma=0.6)
    np.testing.assert_allclose(p.p_exame, np.minimum(1, 0.6 / np.log2(1 + np.arange(1, 6))))


def test_fluxos_reproduziveis():
    p = parametros()
    f1, f2 = Fluxos(7, p), Fluxos(7, p)
    np.testing.assert_array_equal(f1.n_entradas, f2.n_entradas)
    np.testing.assert_array_equal(f1.u_saida, f2.u_saida)
    np.testing.assert_array_equal(f1.gerador_lote(2, 1, 0).random(5), f2.gerador_lote(2, 1, 0).random(5))
    assert not np.array_equal(Fluxos(8, p).u_saida, f1.u_saida)


def test_simulacao_deterministica():
    p = parametros()
    for politica in "ABCD":
        e1 = motor.simular(p, Fluxos(3, p), politica)
        e2 = motor.simular(p, Fluxos(3, p), politica)
        np.testing.assert_array_equal(e1.H, e2.H)
        np.testing.assert_array_equal(e1.n_aval, e2.n_aval)


def test_entradas_e_saidas_iguais_entre_politicas():
    p = parametros()
    estados = [motor.simular(p, Fluxos(5, p), politica) for politica in "ABCD"]
    for est in estados[1:]:
        np.testing.assert_array_equal(est.entrada, estados[0].entrada)
        np.testing.assert_array_equal(est.saida, estados[0].saida)


def test_capacidade_nunca_excedida():
    p = parametros(clientes_por_rodada=400, compatibilidade=1.0)
    fl = Fluxos(1, p)
    est = motor.inicializar(p, fl)
    for t in range(p.T_total):
        antes = est.H.copy()
        motor.rodada(est, t, "B", p, fl, coletar=False)
        assert (est.H - antes).max() <= p.capacidade


def test_exames_maiores_ou_iguais_contratacoes():
    p = parametros()
    est = motor.simular(p, Fluxos(2, p), "C")
    assert (est.E >= est.H).all()


def test_ordem_de_ocorrencia():
    np.testing.assert_array_equal(ordem_de_ocorrencia(np.array([5, 3, 5, 5, 3])), [0, 0, 1, 2, 1])


def test_primeiros_k():
    eleg = np.array([[True, False, True, True], [False, False, False, True]])
    np.testing.assert_array_equal(primeiros_k(eleg, 2), [[0, 2], [3, -1]])


def test_cota_reserva_posicoes_sem_rebaixar_novato():
    p = parametros()
    fl = Fluxos(1, p)
    est = motor.inicializar(p, fl)
    t = 10
    cand = np.arange(8)
    est.n_aval[cand] = 10
    est.soma_notas[cand] = 10 * np.array([5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3])
    est.entrada[cand] = -50
    est.entrada[[0, 6, 7]] = t - 1           # novatos: o melhor (0) e dois piores (6, 7)
    elegivel = np.ones((1, len(cand)), dtype=bool)
    lista = listas_cota(est, cand, elegivel, t, p)[0]
    assert lista[0] == 0                     # novato bem ranqueado não é rebaixado
    assert lista[1] == 6 and lista[3] == 7   # posições 2 e 4 reservadas aos outros novatos
    assert lista[2] == 1                     # demais posições seguem a ordem de B


def test_kaplan_meier_e_rmst():
    h = 4
    todos_na_1a = metricas.kaplan_meier(np.ones(10, int), np.ones(10, bool), h)
    assert metricas.rmst(todos_na_1a, h) == pytest.approx(1.0)
    nenhum = metricas.kaplan_meier(np.full(10, h), np.zeros(10, bool), h)
    assert metricas.rmst(nenhum, h) == pytest.approx(h)


def test_gini():
    assert metricas.gini(np.ones(5)) == pytest.approx(0.0)
    assert metricas.gini(np.array([0, 0, 0, 1.0])) == pytest.approx(0.75)


def test_ramificacao_pareada_gera_metricas():
    p = parametros()
    linhas = rodar_semente((p, 11, ("A", "B", "C", "D")))
    assert [l["politica"] for l in linhas] == ["A", "B", "C", "D"]
    for linha in linhas:
        assert 0 <= linha["ndcg10"] <= 1
        assert 0 <= linha["rmst_alta"] <= p.horizonte_h
    # mesma semente, mesmo resultado
    assert rodar_semente((p, 11, ("B",)))[0] == linhas[1]


def test_aquecimento_usa_politica_de_referencia():
    from coldstart.experimento import momentos_referencia
    p_b, p_d = parametros(politica_referencia="B"), parametros(politica_referencia="D")
    # mesmos fluxos, políticas diferentes no aquecimento: estados podem divergir,
    # mas cada um é determinístico
    m1, m2 = momentos_referencia((p_d, 3)), momentos_referencia((p_d, 3))
    assert m1.keys() == m2.keys()
    for chave in m1:
        np.testing.assert_equal(m1[chave], m2[chave])   # trata NaN == NaN
    assert rodar_semente((p_b, 3, ("A",)))[0] == rodar_semente((p_b, 3, ("A",)))[0]


def test_momentos_simulados_no_intervalo():
    p = parametros()
    m = metricas.momentos_simulados(motor.simular(p, Fluxos(4, p), "B"), p)
    assert m["avaliacoes_por_fornecedor_mes"] >= 0
