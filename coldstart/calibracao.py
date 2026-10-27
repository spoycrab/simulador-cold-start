"""Calibração do simulador a partir dos dados do Inside Airbnb.

Esta etapa produz tudo o que os dados permitem estimar diretamente:

* população de fornecedores e categorias de consulta (faixas de preço);
* distribuição da qualidade latente (duas escalas, ver README);
* taxa recente de entrada de fornecedores;
* alvos de calibração, que serão ajustados na etapa 2 (u0 e demanda);
* alvos de validação, que NÃO entram no ajuste e servem para testar o
  simulador depois de calibrado;
* amostra estratificada de fornecedores para a simulação.

Parâmetros que os dados não identificam (gamma, tau, p_review, sigma,
churn) permanecem como hipóteses declaradas no arquivo de parâmetros.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import __version__
from . import dados as mod_dados
from .fornecedores import construir_fornecedores, coorte_observada, sem_avaliacao_na_janela
from .saidas import salvar_tabela

NOTA_MIN, NOTA_MAX = 1.0, 5.0


# ---------------------------------------------------------------------------
# Categorias de consulta
# ---------------------------------------------------------------------------

def filtrar_por_anuncios(fornecedores: pd.DataFrame, maximo: int | None) -> pd.DataFrame:
    """Mantém anfitriões com no máximo `maximo` anúncios (None = todos).

    Com maximo = 1, fornecedor e item ranqueado coincidem, e anfitriões com
    muitos anúncios (que operam como empresas) não distorcem a demanda.
    """
    if maximo is None:
        return fornecedores
    return fornecedores[fornecedores["n_anuncios"] <= maximo]


def atribuir_faixas_preco(fornecedores: pd.DataFrame, n_faixas: int) -> tuple[pd.DataFrame, list[float]]:
    """Divide os fornecedores com preço em `n_faixas` quantis do preço mediano.

    Fornecedores sem preço em todos os anúncios são excluídos e contados.
    Retorna a tabela filtrada (com a coluna faixa_preco) e os limites das faixas.
    """
    com_preco = fornecedores[fornecedores["preco_mediano"].notna()].copy()
    faixas, limites = pd.qcut(com_preco["preco_mediano"], n_faixas,
                              labels=False, retbins=True, duplicates="drop")
    if faixas.nunique() < n_faixas:
        raise ValueError("Não foi possível formar faixas de preço distintas; reduza n_faixas_preco.")
    com_preco["faixa_preco"] = faixas.astype(int)
    return com_preco, [float(x) for x in limites]


# ---------------------------------------------------------------------------
# Qualidade latente
# ---------------------------------------------------------------------------

def nota_para_unitario(nota) -> np.ndarray | float:
    """Converte a nota de 1 a 5 para a escala [0, 1]."""
    return (np.asarray(nota, dtype=float) - NOTA_MIN) / (NOTA_MAX - NOTA_MIN)


def qualidade_percentil(notas: pd.Series) -> pd.Series:
    """Qualidade como percentil (posto médio) da nota entre os avaliados.

    Preserva a ordem das notas e distribui a qualidade uniformemente em (0, 1),
    evitando que a compressão das notas no topo comprima o NDCG.
    """
    validas = notas.dropna()
    postos = validas.rank(method="average")
    return ((postos - 0.5) / len(validas)).reindex(notas.index)


def ajustar_beta_qualidade(fornecedores: pd.DataFrame, sigma: float, n_min: int) -> dict:
    """Ajusta Beta(a, b) à qualidade na escala da nota, descontando o ruído.

    Método dos momentos: Var(q) = Var(nota observada) - sigma^2 * E[1/n],
    usando apenas fornecedores com pelo menos `n_min` avaliações.
    """
    base = fornecedores[(fornecedores["n_avaliacoes"] >= n_min) & fornecedores["nota_media"].notna()]
    x = nota_para_unitario(base["nota_media"])
    media = float(np.mean(x))
    var_obs = float(np.var(x, ddof=1))
    var_ruido = float(sigma ** 2 * np.mean(1.0 / base["n_avaliacoes"]))
    var_q = max(var_obs - var_ruido, 1e-6)
    limite = media * (1 - media)
    if var_q >= limite:
        raise ValueError("Variância incompatível com uma distribuição Beta.")
    fator = limite / var_q - 1
    return {
        "a": media * fator, "b": (1 - media) * fator,
        "media": media, "var_observada": var_obs,
        "var_ruido_descontada": var_ruido, "var_qualidade": var_q,
        "n_fornecedores_usados": int(len(base)), "n_min_avaliacoes": n_min,
    }


def quantis_nota(notas: pd.Series, n_pontos: int = 101) -> list[float]:
    """Função quantil empírica da nota, para mapear qualidade em nota (escala percentil)."""
    return [float(v) for v in notas.dropna().quantile(np.linspace(0, 1, n_pontos))]


# ---------------------------------------------------------------------------
# Entrada e alvos
# ---------------------------------------------------------------------------

def taxa_entrada_recente(fornecedores: pd.DataFrame, ano: int) -> dict:
    """Taxa mensal de entrada no ano de referência, entre os sobreviventes."""
    n_ano = int((fornecedores["ano_entrada"] == ano).sum())
    n_total = int(len(fornecedores))
    return {
        "ano_referencia": ano, "entradas_no_ano": n_ano,
        "fornecedores_na_coleta": n_total,
        "taxa_mensal": n_ano / 12 / n_total,
        "observacao": "Calculada apenas com fornecedores ativos na coleta (viés de sobrevivência).",
    }


def alvos_calibracao(fornecedores: pd.DataFrame, janela: int, excluir_teto: bool,
                     p_review: float) -> dict:
    """Estatísticas usadas para ajustar u0 e a demanda na etapa 2."""
    coorte = coorte_observada(fornecedores, janela, excluir_teto)
    sem = sem_avaliacao_na_janela(coorte, janela)
    maduros = fornecedores[fornecedores["meses_cadastro"] >= 12]
    aval_mes = float(maduros["n_avaliacoes_12m"].mean() / 12)
    aval_mes_mediana = float(maduros["n_avaliacoes_12m"].median() / 12)
    return {
        "prop_sem_avaliacao_janela": {
            "valor": float(sem.mean()), "janela_meses": janela,
            "n_coorte": int(len(coorte)),
            "comparavel": False,
            "uso": "referência; não usado no ajuste",
            "observacao": "inflada pela diferença entre idade da conta e idade do anúncio",
        },
        "avaliacoes_por_fornecedor_mes": {
            "valor": aval_mes, "mediana": aval_mes_mediana,
            "n_fornecedores": int(len(maduros)),
            "uso": "ajuste da demanda (clientes por rodada)",
        },
        "contratacoes_por_fornecedor_mes": {
            "valor": aval_mes / p_review, "p_review_assumido": p_review,
            "uso": "derivado: avaliações por mês divididas por p_review (hipótese)",
        },
    }


def alvos_validacao(fornecedores: pd.DataFrame, janela: int, excluir_teto: bool) -> dict:
    """Estatísticas da coorte de 12 a 23 meses e estatísticas de referência.

    Na coorte de 12 a 23 meses, a idade da conta e a do anúncio atual
    praticamente coincidem. Em contas mais antigas, o anúncio atual pode ter
    sido criado anos depois da conta, o que infla as medidas contadas a partir
    do cadastro (ver a tabela avaliacao_por_idade_conta da auditoria). Essas
    medidas ficam registradas como referência não comparável.
    """
    coorte = coorte_observada(fornecedores, janela, excluir_teto)
    sem = sem_avaliacao_na_janela(coorte, janela)
    tempos = coorte.loc[~sem, "meses_ate_1a_avaliacao"]
    tempos = tempos[tempos >= 0]
    jovem = fornecedores[(fornecedores["meses_cadastro"] >= 12) & (fornecedores["meses_cadastro"] < 24)]
    meses_1a = jovem["meses_ate_1a_avaliacao"].clip(lower=0)   # diferenças de dias no mesmo mês
    return {
        "prop_sem_avaliacao_coorte_12_23m": {
            "valor": float((jovem["n_avaliacoes"] == 0).mean()), "n": int(len(jovem)),
            "definicao": "coorte de 12 a 23 meses sem nenhuma avaliação",
        },
        "mediana_avaliacoes_coorte_12_23m": {
            "valor": float(jovem["n_avaliacoes"].median()), "n": int(len(jovem)),
        },
        "prop_1a_avaliacao_ate_3m_coorte_12_23m": {
            "valor": float((meses_1a <= 3).sum() / len(jovem)), "n": int(len(jovem)),
            "definicao": "coorte de 12 a 23 meses com 1ª avaliação até 3 meses após o cadastro",
        },
        "mediana_meses_ate_1a_avaliacao_coorte_12_23m": {
            "valor": float(meses_1a.dropna().median()), "n": int(meses_1a.notna().sum()),
            "definicao": "avaliados da coorte de 12 a 23 meses",
        },
        "mediana_meses_ate_1a_avaliacao": {
            "valor": float(tempos.median()), "n": int(len(tempos)),
            "definicao": f"coorte com pelo menos {janela} meses, avaliados na janela",
            "comparavel": False,
            "observacao": "inflada pela diferença entre idade da conta e idade do anúncio",
        },
    }


# ---------------------------------------------------------------------------
# Amostra estratificada
# ---------------------------------------------------------------------------

def alocacao_proporcional(contagens: pd.Series, n: int) -> pd.Series:
    """Distribui n unidades entre estratos pelo método dos maiores restos."""
    if n > contagens.sum():
        raise ValueError(
            f"Amostra ({n}) maior que a população ({int(contagens.sum())}). "
            "Reduza projeto.n_fornecedores_amostra no arquivo de parâmetros."
        )
    cota = contagens / contagens.sum() * n
    base = np.floor(cota).astype(int)
    faltam = n - base.sum()
    restos = (cota - base).sort_values(ascending=False)
    base.loc[restos.index[:faltam]] += 1
    return base


def amostra_estratificada(fornecedores: pd.DataFrame, n: int, semente: int,
                          estratos: tuple[str, ...] = ("faixa_preco", "sem_nota")) -> pd.DataFrame:
    """Amostra sem reposição, proporcional ao tamanho de cada estrato.

    Por padrão, estratifica por faixa de preço e pela ausência de nota, para
    preservar a proporção de fornecedores sem histórico (o grupo de interesse).
    """
    gerador = np.random.default_rng(semente)
    chave = fornecedores[list(estratos)].astype(str).agg("|".join, axis=1)
    tamanhos = alocacao_proporcional(chave.value_counts().sort_index(), n)
    partes = []
    for valor, tamanho in tamanhos.items():
        ids = np.sort(fornecedores.index[chave == valor].to_numpy())
        escolhidos = gerador.choice(ids, size=int(tamanho), replace=False)
        partes.append(fornecedores.loc[np.sort(escolhidos)])
    return pd.concat(partes).sort_index()


def comparar_amostra(amostra: pd.DataFrame, populacao: pd.DataFrame) -> pd.DataFrame:
    """Compara amostra e população (média, mediana e teste KS de duas amostras)."""
    variaveis = {
        "nota_media": "Nota média", "n_avaliacoes": "Avaliações",
        "n_avaliacoes_12m": "Avaliações em 12 meses",
        "meses_cadastro": "Meses de cadastro", "preco_mediano": "Preço mediano",
    }
    linhas = []
    for coluna, nome in variaveis.items():
        p = populacao[coluna].dropna()
        a = amostra[coluna].dropna()
        ks = stats.ks_2samp(a, p)
        linhas.append({
            "Variável": nome,
            "Média (população)": float(p.mean()), "Média (amostra)": float(a.mean()),
            "Mediana (população)": float(p.median()), "Mediana (amostra)": float(a.median()),
            "KS": float(ks.statistic), "p-valor KS": float(ks.pvalue),
        })
    linhas.append({
        "Variável": "Sem nota (%)",
        "Média (população)": 100 * float(populacao["sem_nota"].mean()),
        "Média (amostra)": 100 * float(amostra["sem_nota"].mean()),
        "Mediana (população)": float("nan"), "Mediana (amostra)": float("nan"),
        "KS": float("nan"), "p-valor KS": float("nan"),
    })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def _para_json(objeto):
    if isinstance(objeto, (np.integer,)):
        return int(objeto)
    if isinstance(objeto, (np.floating,)):
        return float(objeto)
    if isinstance(objeto, pd.Timestamp):
        return str(objeto.date())
    raise TypeError(f"Tipo não serializável: {type(objeto)}")


def executar_calibracao(config: dict, pasta_dados: str | Path, pasta_saida: str | Path) -> dict:
    """Executa a calibração completa e grava os resultados em `pasta_saida`."""
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    cal, proj = config["calibracao"], config["projeto"]

    arquivos = mod_dados.localizar_arquivos(pasta_dados)
    manifesto = mod_dados.gerar_manifesto(arquivos, pasta_saida / "manifesto_dados.json")
    listings = mod_dados.ler_listings(arquivos["listings"])
    data_coleta = (pd.Timestamp(config["dados"]["data_coleta"])
                   if config["dados"]["data_coleta"] else mod_dados.data_da_coleta(listings))

    todos = construir_fornecedores(listings, data_coleta)
    elegiveis = filtrar_por_anuncios(todos, cal.get("max_anuncios_por_fornecedor"))
    populacao, limites = atribuir_faixas_preco(elegiveis, cal["n_faixas_preco"])
    populacao["q_percentil"] = qualidade_percentil(populacao["nota_media"])
    populacao["q_nota"] = nota_para_unitario(populacao["nota_media"])

    beta = ajustar_beta_qualidade(populacao, proj["sigma_ruido"], cal["n_min_avaliacoes_beta"])
    # Entrada e alvos usam a mesma população da simulação (fornecedores com preço)
    entrada = taxa_entrada_recente(populacao, cal["ano_referencia_entrada"])
    alvos_cal = alvos_calibracao(populacao, cal["janela_sem_avaliacao_meses"],
                                 cal["excluir_teto_15_anos"], proj["p_review"])
    alvos_val = alvos_validacao(populacao, cal["janela_sem_avaliacao_meses"],
                                cal["excluir_teto_15_anos"])

    amostra = amostra_estratificada(populacao, proj["n_fornecedores_amostra"], proj["semente_amostra"])
    colunas_amostra = ["faixa_preco", "n_anuncios", "preco_mediano", "nota_media", "sem_nota",
                       "n_avaliacoes", "n_avaliacoes_12m", "meses_cadastro",
                       "q_percentil", "q_nota"]
    amostra[colunas_amostra].to_csv(pasta_saida / "fornecedores_amostra.csv",
                                    index_label="host_id", encoding="utf-8")

    resultado = {
        "versao_codigo": __version__,
        "data_coleta": data_coleta,
        "arquivos": manifesto["arquivos"],
        "populacao": {
            "fornecedores_total": int(len(todos)),
            "max_anuncios_por_fornecedor": cal.get("max_anuncios_por_fornecedor"),
            "fornecedores_elegiveis": int(len(elegiveis)),
            "excluidos_sem_preco": int(len(elegiveis) - len(populacao)),
            "fornecedores_com_preco": int(len(populacao)),
            "n_faixas_preco": cal["n_faixas_preco"],
            "limites_faixas_preco": limites,
            "fornecedores_por_faixa": {int(k): int(v) for k, v in
                                       populacao["faixa_preco"].value_counts().sort_index().items()},
        },
        "nota": {
            "media_global": float(populacao["nota_media"].mean()),
            "quantis": quantis_nota(populacao["nota_media"]),
        },
        "qualidade": {"escala_configurada": proj["escala_qualidade"], "beta_escala_nota": beta},
        "entrada": entrada,
        "alvos_calibracao": alvos_cal,
        "alvos_validacao": alvos_val,
        "amostra": {
            "arquivo": "fornecedores_amostra.csv",
            "n": int(len(amostra)), "semente": proj["semente_amostra"],
            "por_faixa": {int(k): int(v) for k, v in
                          amostra["faixa_preco"].value_counts().sort_index().items()},
        },
        "hipoteses_nao_identificadas": {
            "gamma": proj["escolha"]["gamma"], "tau": proj["escolha"]["tau"],
            "p_review": proj["p_review"], "sigma_ruido": proj["sigma_ruido"],
            "churn_mensal": proj["churn_mensal"],
        },
    }
    (pasta_saida / "calibracao.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=_para_json), encoding="utf-8")

    _gravar_tabelas(resultado, comparar_amostra(amostra, populacao), pasta_saida)
    return resultado


def _gravar_tabelas(res: dict, comparacao: pd.DataFrame, pasta: Path) -> None:
    beta = res["qualidade"]["beta_escala_nota"]
    entrada = res["entrada"]
    parametros = pd.DataFrame([
        {"Parâmetro": "Anfitriões na coleta", "Valor": res["populacao"]["fornecedores_total"],
         "Origem": "dados"},
        {"Parâmetro": "Anfitriões elegíveis (limite de anúncios)",
         "Valor": res["populacao"]["fornecedores_elegiveis"], "Origem": "dados"},
        {"Parâmetro": "Fornecedores (população com preço)", "Valor": res["populacao"]["fornecedores_com_preco"],
         "Origem": "dados"},
        {"Parâmetro": "Fornecedores excluídos sem preço", "Valor": res["populacao"]["excluidos_sem_preco"],
         "Origem": "dados"},
        {"Parâmetro": "Nota média global (m)", "Valor": res["nota"]["media_global"], "Origem": "dados"},
        {"Parâmetro": "Beta da qualidade: a", "Valor": beta["a"], "Origem": "calibrado"},
        {"Parâmetro": "Beta da qualidade: b", "Valor": beta["b"], "Origem": "calibrado"},
        {"Parâmetro": f"Taxa mensal de entrada ({entrada['ano_referencia']}, %)",
         "Valor": 100 * entrada["taxa_mensal"], "Origem": "calibrado"},
    ])
    for i, limite in enumerate(res["populacao"]["limites_faixas_preco"][1:-1], start=1):
        parametros.loc[len(parametros)] = {"Parâmetro": f"Limite entre faixas {i - 1} e {i} (preço)",
                                           "Valor": limite, "Origem": "dados"}
    salvar_tabela(parametros, pasta, "parametros_calibrados",
                  "Parâmetros estimados a partir dos dados", "tab:parametros_calibrados")

    linhas = []
    for nome, item in res["alvos_calibracao"].items():
        linhas.append({"Alvo": nome, "Valor": item["valor"], "Papel": "calibração"})
    for nome, item in res["alvos_validacao"].items():
        linhas.append({"Alvo": nome, "Valor": item["valor"], "Papel": "validação"})
    salvar_tabela(pd.DataFrame(linhas), pasta, "alvos",
                  "Alvos de calibração e de validação do simulador", "tab:alvos", casas=3)
    salvar_tabela(comparacao, pasta, "comparacao_amostra",
                  "Comparação entre a amostra estratificada e a população", "tab:comparacao_amostra")
