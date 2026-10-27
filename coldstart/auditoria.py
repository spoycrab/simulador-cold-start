"""Auditoria reproduzível dos dados do Inside Airbnb.

Recalcula, a partir dos arquivos brutos, todas as estatísticas descritivas
citadas no texto e as verificações de consistência entre arquivos. O
objetivo é que nenhum número do artigo dependa de transcrição manual.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import dados as mod_dados
from .fornecedores import (
    DIAS_POR_MES, concentracao_topo, construir_fornecedores, coorte_observada,
    sem_avaliacao_na_janela,
)
from .saidas import salvar_tabela


class Registro:
    """Acumula linhas da tabela de auditoria."""

    def __init__(self) -> None:
        self.linhas: list[dict] = []

    def add(self, grupo: str, indicador: str, valor, observacao: str = "") -> None:
        self.linhas.append({
            "Grupo": grupo, "Indicador": indicador,
            "Valor": valor, "Observação": observacao,
        })

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame(self.linhas)


def auditar_listings(listings: pd.DataFrame, fornecedores: pd.DataFrame,
                     data_coleta: pd.Timestamp, reg: Registro,
                     ano_referencia: int, janela_meses: int) -> None:
    g = "Coleta"
    reg.add(g, "Data da coleta (scrape_id)", str(data_coleta.date()))
    datas = listings["last_scraped"].dt.date.value_counts().sort_index()
    for data, n in datas.items():
        reg.add(g, f"Anúncios atualizados em {data}", int(n), "last_scraped")
    reg.add(g, "Anúncios", int(listings["id"].nunique()))
    reg.add(g, "Fornecedores (anfitriões)", int(len(fornecedores)))
    reg.add(g, "host_since vazio (%)", 100 * listings["host_since"].isna().mean())

    g = "Fornecedores"
    n_aval = fornecedores["n_avaliacoes"]
    notas = fornecedores["nota_media"].dropna()
    reg.add(g, "Sem nenhuma avaliação (%)", 100 * (n_aval == 0).mean())
    reg.add(g, "Com até três avaliações (%)", 100 * (n_aval <= 3).mean())
    reg.add(g, "Mediana de avaliações", float(n_aval.median()))
    reg.add(g, "Nota média", float(notas.mean()), "média das notas dos anúncios")
    reg.add(g, "Desvio padrão da nota", float(notas.std()))
    reg.add(g, "Mediana da nota", float(notas.median()))
    reg.add(g, "Nota maior ou igual a 4,8 (%)", 100 * (notas >= 4.8).mean())
    reg.add(g, "Avaliações nos 10% maiores (%)", 100 * concentracao_topo(n_aval, 0.10))
    reg.add(g, "Sem preço em todos os anúncios", int(fornecedores["preco_mediano"].isna().sum()))
    unico = fornecedores[fornecedores["n_anuncios"] == 1]
    reg.add(g, "Com um único anúncio", int(len(unico)), "população padrão da simulação")
    reg.add(g, "Com um único anúncio e preço", int(unico["preco_mediano"].notna().sum()))
    reg.add(g, "Avaliações por mês, últimos 12 meses: média", float(
        fornecedores.loc[fornecedores["meses_cadastro"] >= 12, "n_avaliacoes_12m"].mean() / 12),
        "todos os anfitriões com 12 meses ou mais")
    reg.add(g, "Avaliações por mês, últimos 12 meses: mediana", float(
        fornecedores.loc[fornecedores["meses_cadastro"] >= 12, "n_avaliacoes_12m"].median() / 12))

    g = "Reconstrução da entrada"
    reg.add(g, "Fornecedores no teto de 15 anos", int(fornecedores["teto_cadastro"].sum()),
            "hosts_time_as_host_years = 15")
    reg.add(g, "Tempo de cadastro divergente entre anúncios",
            int(fornecedores["tempo_divergente"].sum()))
    reg.add(g, "Entrada mais antiga reconstruída",
            str(fornecedores["data_entrada"].min().date()))
    avaliados = fornecedores[fornecedores["primeira_avaliacao"].notna()]
    precede_forn = (avaliados["data_entrada"] <= avaliados["primeira_avaliacao"]).mean()
    reg.add(g, "Cadastro antes da 1ª avaliação, por fornecedor (%)", 100 * precede_forn)
    anuncios = listings.dropna(subset=["first_review"])
    entrada_anuncio = anuncios["host_id"].map(fornecedores["data_entrada"])
    precede_anuncio = (entrada_anuncio <= anuncios["first_review"]).mean()
    reg.add(g, "Cadastro antes da 1ª avaliação, por anúncio (%)", 100 * precede_anuncio,
            "o texto original afirmava 99,5%")

    g = "Entrada de fornecedores"
    meses_max = fornecedores["meses_cadastro"].max()
    media_hist = len(fornecedores) / meses_max
    reg.add(g, "Média histórica de entradas por mês", media_hist,
            "sobreviventes / meses desde a entrada mais antiga")
    reg.add(g, "Taxa histórica mensal (%)", 100 * media_hist / len(fornecedores))
    n_ref = int((fornecedores["ano_entrada"] == ano_referencia).sum())
    reg.add(g, f"Entradas em {ano_referencia} (sobreviventes)", n_ref)
    reg.add(g, f"Taxa mensal em {ano_referencia} (%)", 100 * n_ref / 12 / len(fornecedores))

    g = "Primeira avaliação"
    coorte = coorte_observada(fornecedores, janela_meses, excluir_teto=True)
    sem = sem_avaliacao_na_janela(coorte, janela_meses)
    reg.add(g, f"Sem avaliação nos primeiros {janela_meses} meses (%)", 100 * sem.mean(),
            f"coorte com pelo menos {janela_meses} meses, sem teto")
    tempos = coorte.loc[~sem, "meses_ate_1a_avaliacao"]
    tempos = tempos[tempos >= 0]
    reg.add(g, "Mediana até a 1ª avaliação (dias)", float(tempos.median() * DIAS_POR_MES),
            "avaliados na janela, intervalo não negativo")
    original = coorte_observada(fornecedores, janela_meses, excluir_teto=False)
    tempos_orig = original["meses_ate_1a_avaliacao"].dropna()
    reg.add(g, "Mediana até a 1ª avaliação, definição do texto original (dias)",
            float(tempos_orig.median() * DIAS_POR_MES),
            "todos os avaliados da coorte; o texto original citava 88 dias")


def auditar_reviews(listings: pd.DataFrame, reviews: pd.DataFrame, reg: Registro) -> None:
    g = "Arquivo reviews"
    reg.add(g, "Avaliações", int(len(reviews)))
    reg.add(g, "Período", f"{reviews['date'].min().date()} a {reviews['date'].max().date()}")
    primeira = reviews.groupby("listing_id")["date"].min()
    contagem = reviews.groupby("listing_id").size()
    base = listings.set_index("id")
    comuns = base.index.intersection(primeira.index)
    iguais = (base.loc[comuns, "first_review"] == primeira.loc[comuns]).mean()
    reg.add(g, "first_review igual à 1ª data em reviews (%)", 100 * iguais)
    n_arquivo = contagem.reindex(base.index, fill_value=0)
    reg.add(g, "number_of_reviews igual à contagem em reviews (%)",
            100 * (base["number_of_reviews"] == n_arquivo).mean())


def auditar_calendar(listings: pd.DataFrame, calendar: pd.DataFrame,
                     data_coleta: pd.Timestamp, reg: Registro) -> None:
    g = "Arquivo calendar"
    reg.add(g, "Janela", f"{calendar['date'].min().date()} a {calendar['date'].max().date()}")
    ids_cal = set(calendar["listing_id"].unique())
    ids_lst = set(listings["id"].unique())
    reg.add(g, "Anúncios de listings cobertos (%)", 100 * len(ids_cal & ids_lst) / len(ids_lst))
    reg.add(g, "Anúncios só no calendário", len(ids_cal - ids_lst))
    distancia = abs((calendar["date"].min() - data_coleta).days)
    reg.add(g, "Distância entre início e data da coleta (dias)", int(distancia),
            "deve ser 0 ou próximo de 0")
    indisponivel = calendar["available"].astype(str) == "f"
    reg.add(g, "Dias indisponíveis (%)", 100 * indisponivel.mean(),
            "inclui reservas e bloqueios do anfitrião")
    por_anuncio = indisponivel.groupby(calendar["listing_id"]).mean()
    reg.add(g, "Mediana de dias indisponíveis por anúncio (%)", 100 * por_anuncio.median())


def executar_auditoria(config: dict, pasta_dados: str | Path, pasta_saida: str | Path) -> pd.DataFrame:
    """Lê os arquivos, grava o manifesto e as tabelas de auditoria."""
    pasta_saida = Path(pasta_saida)
    arquivos = mod_dados.localizar_arquivos(pasta_dados)
    print("Arquivos encontrados:")
    for tipo, caminho in arquivos.items():
        print(f"  {tipo:9s} {caminho.name if caminho else '(ausente)'}")

    print("Calculando SHA-256 dos arquivos...")
    mod_dados.gerar_manifesto(arquivos, pasta_saida / "manifesto_dados.json")

    listings = mod_dados.ler_listings(arquivos["listings"])
    data_coleta = (pd.Timestamp(config["dados"]["data_coleta"])
                   if config["dados"]["data_coleta"] else mod_dados.data_da_coleta(listings))
    fornecedores = construir_fornecedores(listings, data_coleta)

    reg = Registro()
    cal = config["calibracao"]
    auditar_listings(listings, fornecedores, data_coleta, reg,
                     cal["ano_referencia_entrada"], cal["janela_sem_avaliacao_meses"])
    if arquivos["reviews"] is not None:
        print("Lendo reviews...")
        auditar_reviews(listings, mod_dados.ler_reviews(arquivos["reviews"]), reg)
    if arquivos["calendar"] is not None:
        print("Lendo calendar...")
        auditar_calendar(listings, mod_dados.ler_calendar(arquivos["calendar"]), data_coleta, reg)

    tabela = reg.tabela()
    salvar_tabela(tabela, pasta_saida, "auditoria",
                  "Auditoria dos dados do Inside Airbnb (São Paulo)", "tab:auditoria")
    entradas = (fornecedores.groupby("ano_entrada").size()
                .rename("fornecedores").reset_index()
                .rename(columns={"ano_entrada": "Ano de entrada", "fornecedores": "Fornecedores"}))
    salvar_tabela(entradas, pasta_saida, "entradas_por_ano",
                  "Fornecedores ativos na coleta, por ano de entrada reconstruído",
                  "tab:entradas_por_ano", casas=0)
    salvar_tabela(avaliacao_por_idade_conta(fornecedores, cal), pasta_saida,
                  "avaliacao_por_idade_conta",
                  "Avaliações por idade da conta (anfitriões com um único anúncio)",
                  "tab:avaliacao_por_idade_conta", casas=3)
    return tabela


def avaliacao_por_idade_conta(fornecedores: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Mostra que medidas contadas desde o cadastro crescem com a idade da conta.

    A fração sem nenhuma avaliação é estável, mas a fração sem avaliação nos
    primeiros 24 meses e o tempo até a 1ª avaliação crescem com a idade da
    conta: em contas antigas, o anúncio atual foi criado depois da conta.
    """
    base = fornecedores
    maximo = cal.get("max_anuncios_por_fornecedor")
    if maximo is not None:
        base = base[base["n_anuncios"] <= maximo]
    base = base[base["preco_mediano"].notna()].copy()
    base["anos_conta"] = (base["meses_cadastro"] // 12).astype(int)
    base["sem_24m"] = sem_avaliacao_na_janela(base, 24)
    linhas = []
    for anos, grupo in base[base["anos_conta"] <= 10].groupby("anos_conta"):
        meses = grupo["meses_ate_1a_avaliacao"].clip(lower=0).dropna()
        linhas.append({
            "Idade da conta (anos)": int(anos), "Fornecedores": int(len(grupo)),
            "Sem nenhuma avaliação": float((grupo["n_avaliacoes"] == 0).mean()),
            "Sem avaliação nos 1ºs 24 meses": float(grupo["sem_24m"].mean()) if anos >= 2 else float("nan"),
            "Mediana de meses até a 1ª avaliação": float(meses.median()),
        })
    return pd.DataFrame(linhas)
