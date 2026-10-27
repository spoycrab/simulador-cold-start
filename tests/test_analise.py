"""Testes da etapa 3: estatística, decisões das hipóteses, análise e sensibilidade."""

import json

import numpy as np
import pandas as pd
import pytest

from coldstart import analise, sensibilidade
from coldstart.estatistica import holm, ic_bootstrap, resumo_diferenca, wilcoxon_unilateral
from coldstart.experimento import rodar_semente
from coldstart.configuracao import carregar_config
from test_simulador import parametros


def test_holm():
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_bootstrap_contem_media_e_e_deterministico():
    d = np.random.default_rng(1).normal(0.5, 1, 40)
    inf, sup = ic_bootstrap(d, 0.95, semente=3)
    assert inf < d.mean() < sup
    assert (inf, sup) == ic_bootstrap(d, 0.95, semente=3)
    assert resumo_diferenca(d)["ic90_inf"] > inf   # IC 90% é mais estreito


def test_wilcoxon_respeita_direcao():
    d = -np.abs(np.random.default_rng(2).normal(1, 0.3, 30))
    assert wilcoxon_unilateral(d, "menor") < 0.01
    assert wilcoxon_unilateral(d, "maior") > 0.99
    assert wilcoxon_unilateral(np.zeros(10), "menor") == 1.0


def _experimento_sintetico(n=30):
    rng = np.random.default_rng(0)
    linhas = []
    efeito_rmst = {"A": 1.0, "B": 0.0, "C": -1.0, "D": -0.02}
    efeito_ndcg = {"A": -0.05, "B": 0.0, "C": -0.10, "D": -0.001}
    for s in range(n):
        base_rmst, base_ndcg = rng.normal(10, 1), rng.normal(0.8, 0.01)
        for pol in "ABCD":
            linha = {"semente": s, "politica": pol,
                     "rmst_alta": base_rmst + efeito_rmst[pol] + rng.normal(0, 0.1),
                     "ndcg10": base_ndcg + efeito_ndcg[pol] + rng.normal(0, 0.0005)}
            for m in analise.METRICAS:
                linha.setdefault(m, rng.random())
            linhas.append(linha)
    return pd.DataFrame(linhas)


def test_decisoes_das_hipoteses():
    df = _experimento_sintetico()
    h = analise.decidir_hipoteses(df, [("C", "B"), ("D", "B")], delta=0.02).set_index(
        ["hipotese", "comparacao"])
    assert h.loc[("H1", "C vs B"), "sustentada"]          # C reduz o RMST
    assert not h.loc[("H2", "C vs B"), "sustentada"]      # perda de 0,10 > delta
    assert h.loc[("H2", "D vs B"), "sustentada"]          # perda de 0,001 < delta


def test_analise_completa_gera_tabelas_e_figuras(tmp_path):
    df = _experimento_sintetico()
    pasta = tmp_path / "experimento"
    pasta.mkdir()
    df.to_csv(pasta / "metricas_por_semente.csv", index=False)
    km = [{"semente": s, "politica": p, "t": t, "S": max(0.0, 1 - 0.1 * t)}
          for s in range(30) for p in "ABCD" for t in range(5)]
    pd.DataFrame(km).to_csv(pasta / "km_alta.csv", index=False)
    (pasta / "parametros.json").write_text(json.dumps({"horizonte_h": 4}))
    analise.executar_analise(carregar_config(), tmp_path)
    for nome in ("hipoteses.csv", "hipoteses.tex", "comparacoes.csv", "metricas_por_politica.tex",
                 "nao_inferioridade_por_delta.csv", "fig_kaplan_meier_alta.pdf", "fig_fronteira.pdf"):
        assert (tmp_path / "analise" / nome).exists(), nome


def test_semente_inclui_curva_km():
    p = parametros()
    linha = rodar_semente((p, 1, ("B",)))[0]
    assert len(linha["_km_alta"]) == p.horizonte_h + 1
    assert linha["_km_alta"][0] == 1.0


def test_posicoes_para_rho():
    assert sensibilidade.posicoes_para_rho(0.2, 10, (3, 8)) == (3, 8)
    assert sensibilidade.posicoes_para_rho(0.1, 10, (3, 8)) == (3,)
    assert sensibilidade.posicoes_para_rho(0.3, 10, (3, 8)) == (3, 6, 8)


def test_cenarios_um_parametro_por_vez():
    config = carregar_config()
    central = parametros(gamma=0.6, tau=5.0, u0=2.5, kappa=2.0, rho=0.2, posicoes_reservadas=(3, 8),
                         janela_novidade=12, churn=0.02, p_review=0.7, sigma=0.1, K=10,
                         taxa_entrada=0.015)
    lista = sensibilidade.cenarios(config, central)
    assert lista[0]["nome"] == "central"
    for cenario in lista[1:]:
        atributo, _ = sensibilidade.PARAMETROS[cenario["parametro"]]
        assert getattr(cenario["p"], atributo) != getattr(central, atributo)
        # nenhum outro parâmetro de sensibilidade muda
        outros = [a for k, (a, _) in sensibilidade.PARAMETROS.items() if a != atributo]
        assert all(getattr(cenario["p"], a) == getattr(central, a) for a in outros)
