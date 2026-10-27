"""Testes da política B2 (média posterior Beta, sem sorteio) e da decomposição C - B."""

import numpy as np
import pandas as pd
import pytest

from coldstart import motor
from coldstart.analise import tabela_decomposicao
from coldstart.configuracao import carregar_config, validar_config
from coldstart.experimento import rodar_semente
from coldstart.fluxos import Fluxos
from coldstart.politicas import POLITICAS, POLITICAS_PROTOCOLO, escore_beta_media, priori_beta

from test_simulador import parametros


def test_escore_b2_e_a_media_posterior_de_c():
    p = parametros(kappa=2.0)
    est = motor.inicializar(p, Fluxos(1, p))
    cand = np.arange(4)
    est.E[cand] = [0, 10, 10, 3]
    est.H[cand] = [0, 5, 1, 3]
    a0, b0 = priori_beta(0.2, p)
    np.testing.assert_allclose(escore_beta_media(est, cand, 0.2, p),
                               (a0 + est.H[cand]) / (a0 + b0 + est.E[cand]))


def test_novato_comeca_na_taxa_global():
    p = parametros(kappa=2.0)
    est = motor.inicializar(p, Fluxos(1, p))
    cand = np.arange(3)
    est.E[cand] = 0
    est.H[cand] = 0
    np.testing.assert_allclose(escore_beta_media(est, cand, 0.37, p), 0.37)


def test_b2_deterministica():
    p = parametros()
    assert rodar_semente((p, 4, ("B2",))) == rodar_semente((p, 4, ("B2",)))


def test_incluir_b2_nao_altera_as_demais_politicas():
    p = parametros()
    sem = rodar_semente((p, 9, POLITICAS_PROTOCOLO))
    com = rodar_semente((p, 9, POLITICAS))
    com_sem_b2 = [linha for linha in com if linha["politica"] != "B2"]
    assert com_sem_b2 == sem


def test_decomposicao_soma_exata():
    p = parametros()
    df = pd.DataFrame([linha for s in range(4) for linha in rodar_semente((p, s, ("B", "B2", "C")))])
    tabela = tabela_decomposicao(df)
    for metrica, grupo in tabela.groupby("metrica"):
        d = grupo.set_index("parcela")["diferenca_media"]
        assert d["C - B (total)"] == pytest.approx(d["C - B2 (exploração)"] + d["B2 - B (sinal)"])


def test_config_exige_politicas_das_comparacoes():
    config = carregar_config()
    assert "B2" in config["simulacao"]["politicas"]
    config["simulacao"]["politicas"] = ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        validar_config(config)
