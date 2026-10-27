import copy

import numpy as np
import pandas as pd
import pytest

from coldstart.configuracao import carregar_config, validar_config
from coldstart.saidas import escapar_latex, formatar_numero, tabela_latex


def test_config_padrao_e_valida():
    cfg = carregar_config()
    assert cfg["projeto"]["unidade"] == "fornecedor"
    assert cfg["protocolo"]["delta_nao_inferioridade"] == 0.02


def test_posicoes_reservadas_incompativeis_com_rho():
    cfg = copy.deepcopy(carregar_config())
    cfg["projeto"]["politica_D"]["posicoes_reservadas"] = [3]
    with pytest.raises(ValueError, match="posições reservadas"):
        validar_config(cfg)


def test_horizonte_maior_que_T():
    cfg = copy.deepcopy(carregar_config())
    cfg["projeto"]["horizonte_h"] = cfg["projeto"]["T"] + 1
    with pytest.raises(ValueError, match="horizonte_h"):
        validar_config(cfg)


def test_formatacao_brasileira():
    assert formatar_numero(1234.567, 2) == "1.234,57"
    assert formatar_numero(np.int64(15840)) == "15.840"
    assert formatar_numero(float("nan")) == "--"


def test_escape_latex():
    assert escapar_latex("prop_sem_avaliacao 10%") == r"prop\_sem\_avaliacao 10\%"


def test_tabela_latex_tem_estrutura():
    df = pd.DataFrame({"Indicador": ["a_b"], "Valor": [0.5]})
    texto = tabela_latex(df, "Legenda", "tab:x")
    assert r"\begin{tabular}{lr}" in texto and r"a\_b & 0,50 \\" in texto
