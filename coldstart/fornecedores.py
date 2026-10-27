"""Agregação dos anúncios em fornecedores (anfitriões).

A unidade ranqueada no estudo é o fornecedor. Cada linha do resultado
corresponde a um anfitrião (host_id), com as estatísticas agregadas dos seus
anúncios e a data de entrada reconstruída a partir do tempo de cadastro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAS_POR_MES = 30.4375  # média do calendário gregoriano
TETO_ANOS_CADASTRO = 15  # o campo hosts_time_as_host_years satura neste valor


def construir_fornecedores(listings: pd.DataFrame, data_coleta: pd.Timestamp) -> pd.DataFrame:
    """Agrega os anúncios por anfitrião.

    Reconstrução da entrada: o campo host_since é distribuído vazio nas
    coletas recentes; a data de entrada é estimada subtraindo da data da
    coleta o tempo de cadastro informado (anos e meses). A precisão é mensal.
    """
    df = listings.copy()
    df["_meses_cadastro"] = (
        df["hosts_time_as_host_years"] * 12 + df["hosts_time_as_host_months"]
    )
    grupos = df.groupby("host_id")
    fornecedores = pd.DataFrame({
        "n_anuncios": grupos.size(),
        "meses_cadastro": grupos["_meses_cadastro"].first(),
        "anos_cadastro": grupos["hosts_time_as_host_years"].first(),
        "tempo_divergente": grupos["_meses_cadastro"].nunique() > 1,
        "preco_mediano": grupos["preco"].median(),
        "nota_media": grupos["review_scores_rating"].mean(),
        "n_avaliacoes": grupos["number_of_reviews"].sum(),
        "n_avaliacoes_12m": grupos["number_of_reviews_ltm"].sum(),
        "primeira_avaliacao": grupos["first_review"].min(),
    })
    fornecedores["teto_cadastro"] = fornecedores["anos_cadastro"] >= TETO_ANOS_CADASTRO
    fornecedores["data_entrada"] = data_coleta - pd.to_timedelta(
        fornecedores["meses_cadastro"] * DIAS_POR_MES, unit="D"
    )
    fornecedores["ano_entrada"] = fornecedores["data_entrada"].dt.year
    fornecedores["meses_ate_1a_avaliacao"] = (
        (fornecedores["primeira_avaliacao"] - fornecedores["data_entrada"]).dt.days
        / DIAS_POR_MES
    )
    fornecedores["sem_nota"] = fornecedores["nota_media"].isna()
    return fornecedores


def sem_avaliacao_na_janela(fornecedores: pd.DataFrame, meses: int) -> pd.Series:
    """Indica, para cada fornecedor, se não houve avaliação nos primeiros `meses`."""
    limite = fornecedores["data_entrada"] + pd.to_timedelta(meses * DIAS_POR_MES, unit="D")
    avaliado = fornecedores["primeira_avaliacao"].notna() & (
        fornecedores["primeira_avaliacao"] <= limite
    )
    return ~avaliado


def coorte_observada(fornecedores: pd.DataFrame, meses_minimos: int,
                     excluir_teto: bool = True) -> pd.DataFrame:
    """Fornecedores observados por pelo menos `meses_minimos` desde a entrada."""
    coorte = fornecedores[fornecedores["meses_cadastro"] >= meses_minimos]
    if excluir_teto:
        coorte = coorte[~coorte["teto_cadastro"]]
    return coorte


def concentracao_topo(valores: pd.Series, fracao: float = 0.10) -> float:
    """Parcela do total concentrada nos `fracao` maiores valores."""
    ordenados = np.sort(valores.to_numpy())[::-1]
    n_topo = int(len(ordenados) * fracao)
    total = ordenados.sum()
    return float(ordenados[:n_topo].sum() / total) if total > 0 else float("nan")
