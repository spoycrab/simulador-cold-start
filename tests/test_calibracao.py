import json

import numpy as np
import pandas as pd
import pytest

from coldstart import calibracao as cal
from coldstart.auditoria import executar_auditoria
from coldstart.dados import converter_preco
from coldstart.fornecedores import construir_fornecedores, sem_avaliacao_na_janela

DATA = pd.Timestamp("2026-06-14")


@pytest.fixture
def fornecedores(listings_df):
    df = listings_df.copy()
    for coluna in ("last_scraped", "first_review", "last_review"):
        df[coluna] = pd.to_datetime(df[coluna])
    df["preco"] = converter_preco(df["price"])
    return construir_fornecedores(df, DATA)


def test_agregacao_por_fornecedor(listings_df, fornecedores):
    host = listings_df[listings_df["host_id"] == 1]
    linha = fornecedores.loc[1]
    assert linha["n_anuncios"] == len(host)
    assert linha["n_avaliacoes"] == host["number_of_reviews"].sum()
    assert linha["meses_cadastro"] == (host["hosts_time_as_host_years"].iloc[0] * 12
                                       + host["hosts_time_as_host_months"].iloc[0])


def test_faixas_de_preco_equilibradas(fornecedores):
    populacao, limites = cal.atribuir_faixas_preco(fornecedores, 5)
    contagens = populacao["faixa_preco"].value_counts()
    assert len(contagens) == 5 and len(limites) == 6
    assert contagens.max() - contagens.min() <= 0.05 * len(populacao) + 2


def test_qualidade_percentil_preserva_ordem():
    notas = pd.Series([4.1, 5.0, np.nan, 4.5, 4.5])
    q = cal.qualidade_percentil(notas)
    assert np.isnan(q.iloc[2])
    assert q.iloc[0] < q.iloc[3] == q.iloc[4] < q.iloc[1]
    assert q.dropna().between(0, 1).all()


def test_beta_recupera_parametros_sem_ruido():
    rng = np.random.default_rng(0)
    x = rng.beta(5.0, 2.0, size=200_000)
    df = pd.DataFrame({"nota_media": 1 + 4 * x, "n_avaliacoes": 10**9})
    ajuste = cal.ajustar_beta_qualidade(df, sigma=0.1, n_min=10)
    assert ajuste["a"] == pytest.approx(5.0, rel=0.03)
    assert ajuste["b"] == pytest.approx(2.0, rel=0.03)


def test_alocacao_proporcional_soma_n():
    contagens = pd.Series({"a": 333, "b": 333, "c": 334})
    alocacao = cal.alocacao_proporcional(contagens, 100)
    assert alocacao.sum() == 100
    assert set(alocacao) <= {33, 34}


def test_amostra_reproduzivel_e_proporcional(fornecedores):
    populacao, _ = cal.atribuir_faixas_preco(fornecedores, 5)
    a1 = cal.amostra_estratificada(populacao, 100, semente=7)
    a2 = cal.amostra_estratificada(populacao, 100, semente=7)
    a3 = cal.amostra_estratificada(populacao, 100, semente=8)
    assert list(a1.index) == list(a2.index)
    assert list(a1.index) != list(a3.index)
    assert len(a1) == 100 and a1.index.is_unique
    assert abs(a1["sem_nota"].mean() - populacao["sem_nota"].mean()) < 0.02


def test_sem_avaliacao_na_janela():
    df = pd.DataFrame({
        "data_entrada": pd.to_datetime(["2020-01-01"] * 3),
        "primeira_avaliacao": pd.to_datetime(["2020-03-01", "2023-01-01", None]),
    })
    assert list(sem_avaliacao_na_janela(df, 24)) == [False, True, True]


def test_calibracao_completa_gera_arquivos(config, pasta_dados, tmp_path):
    saida = tmp_path / "calibracao"
    res = cal.executar_calibracao(config, pasta_dados, saida)
    for nome in ("calibracao.json", "fornecedores_amostra.csv", "alvos.csv", "alvos.tex",
                 "parametros_calibrados.tex", "comparacao_amostra.csv", "manifesto_dados.json"):
        assert (saida / nome).exists(), nome
    gravado = json.loads((saida / "calibracao.json").read_text(encoding="utf-8"))
    assert gravado["amostra"]["n"] == 50
    assert 0 <= res["alvos_calibracao"]["prop_sem_avaliacao_janela"]["valor"] <= 1
    assert len(gravado["nota"]["quantis"]) == 101


def test_calibracao_e_deterministica(config, pasta_dados, tmp_path):
    cal.executar_calibracao(config, pasta_dados, tmp_path / "r1")
    cal.executar_calibracao(config, pasta_dados, tmp_path / "r2")
    a = (tmp_path / "r1" / "fornecedores_amostra.csv").read_bytes()
    b = (tmp_path / "r2" / "fornecedores_amostra.csv").read_bytes()
    assert a == b


def test_auditoria_gera_tabelas(config, pasta_dados, tmp_path):
    tabela = executar_auditoria(config, pasta_dados, tmp_path / "auditoria")
    assert (tmp_path / "auditoria" / "auditoria.tex").exists()
    assert "Fornecedores (anfitriões)" in set(tabela["Indicador"])


def test_filtro_anuncio_unico(fornecedores):
    unicos = cal.filtrar_por_anuncios(fornecedores, 1)
    assert (unicos["n_anuncios"] == 1).all()
    assert len(cal.filtrar_por_anuncios(fornecedores, None)) == len(fornecedores)
