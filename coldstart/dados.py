"""Localização, verificação e leitura dos arquivos do Inside Airbnb.

Os arquivos são identificados pelo prefixo do nome (listings, reviews,
calendar), aceitando .csv ou .csv.gz e variações como ``reviews_csv.gz``.
Cada arquivo usado recebe um hash SHA-256 registrado em um manifesto, para
que a calibração seja reproduzível e auditável.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TIPOS = ("listings", "reviews", "calendar")
OBRIGATORIOS = ("listings",)

COLUNAS = {
    "listings": [
        "id", "host_id", "scrape_id", "last_scraped", "host_since",
        "hosts_time_as_host_years", "hosts_time_as_host_months",
        "price", "number_of_reviews", "number_of_reviews_ltm",
        "first_review", "last_review", "review_scores_rating",
        "neighbourhood_cleansed", "room_type",
    ],
    "reviews": ["listing_id", "date"],
    "calendar": ["listing_id", "date", "available"],
}


class ErroDados(Exception):
    """Problema nos arquivos de dados que impede a calibração."""


# ---------------------------------------------------------------------------
# Localização dos arquivos
# ---------------------------------------------------------------------------

def _e_arquivo_de_dados(caminho: Path) -> bool:
    nome = caminho.name.lower()
    return caminho.is_file() and (nome.endswith(".csv") or nome.endswith(".gz"))


def localizar_arquivos(pasta: str | Path, data_coleta: pd.Timestamp | None = None) -> dict:
    """Retorna {tipo: caminho} para os arquivos encontrados na pasta.

    Se houver mais de um arquivo de calendário, é escolhido aquele cuja
    primeira data coincide com a data da coleta (o calendário do Inside
    Airbnb começa no dia da coleta). Para os demais tipos, mais de um
    arquivo é tratado como erro, para evitar misturar coletas.
    """
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise ErroDados(
            f"Pasta de dados não encontrada: {pasta.resolve()}\n"
            "Crie a pasta e coloque nela os arquivos listings, reviews e calendar."
        )
    candidatos = {tipo: [] for tipo in TIPOS}
    for caminho in sorted(pasta.iterdir()):
        if not _e_arquivo_de_dados(caminho):
            continue
        for tipo in TIPOS:
            if caminho.name.lower().startswith(tipo):
                candidatos[tipo].append(caminho)

    arquivos: dict[str, Path | None] = {}
    for tipo in TIPOS:
        lista = candidatos[tipo]
        if not lista:
            if tipo in OBRIGATORIOS:
                raise ErroDados(f"Arquivo '{tipo}' não encontrado em {pasta.resolve()}.")
            arquivos[tipo] = None
        elif len(lista) == 1:
            arquivos[tipo] = lista[0]
        elif tipo == "calendar":
            if data_coleta is None:
                # listings é sempre localizado antes (ordem de TIPOS)
                ids = pd.read_csv(arquivos["listings"], usecols=["scrape_id"])
                data_coleta = data_da_coleta(ids)
            arquivos[tipo] = _escolher_calendario(lista, data_coleta)
        else:
            nomes = ", ".join(c.name for c in lista)
            raise ErroDados(
                f"Mais de um arquivo '{tipo}' em {pasta}: {nomes}. "
                "Mantenha apenas o da mesma coleta dos demais."
            )
    return arquivos


def _escolher_calendario(lista: list[Path], data_coleta: pd.Timestamp) -> Path:
    distancias = []
    for caminho in lista:
        amostra = pd.read_csv(caminho, usecols=["date"], nrows=5000)
        primeira = pd.to_datetime(amostra["date"]).min()
        distancias.append((abs((primeira - data_coleta).days), caminho))
    distancias.sort(key=lambda par: par[0])
    melhor_dist, melhor = distancias[0]
    if melhor_dist > 7:
        raise ErroDados(
            "Nenhum arquivo de calendário começa na data da coleta "
            f"({data_coleta.date()}). Verifique se o calendário é da mesma coleta."
        )
    return melhor


# ---------------------------------------------------------------------------
# Integridade
# ---------------------------------------------------------------------------

def sha256_arquivo(caminho: str | Path, tamanho_bloco: int = 1 << 20) -> str:
    """Calcula o SHA-256 do arquivo sem carregá-lo inteiro na memória."""
    resumo = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(tamanho_bloco), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def gerar_manifesto(arquivos: dict, destino: str | Path) -> dict:
    """Grava um JSON com nome, tamanho e SHA-256 de cada arquivo usado."""
    manifesto = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arquivos": {},
    }
    for tipo, caminho in arquivos.items():
        if caminho is None:
            manifesto["arquivos"][tipo] = None
            continue
        caminho = Path(caminho)
        manifesto["arquivos"][tipo] = {
            "nome": caminho.name,
            "bytes": caminho.stat().st_size,
            "sha256": sha256_arquivo(caminho),
        }
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifesto


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def _verificar_colunas(tipo: str, presentes) -> None:
    faltando = [c for c in COLUNAS[tipo] if c not in set(presentes)]
    if faltando:
        raise ErroDados(f"Colunas ausentes no arquivo '{tipo}': {', '.join(faltando)}.")


def ler_listings(caminho: str | Path) -> pd.DataFrame:
    """Lê o arquivo listings com as colunas necessárias e converte tipos."""
    cabecalho = pd.read_csv(caminho, nrows=0).columns
    _verificar_colunas("listings", cabecalho)
    df = pd.read_csv(caminho, usecols=COLUNAS["listings"], low_memory=False)
    for coluna in ("last_scraped", "first_review", "last_review"):
        df[coluna] = pd.to_datetime(df[coluna], errors="coerce")
    df["preco"] = converter_preco(df["price"])
    return df


def ler_reviews(caminho: str | Path) -> pd.DataFrame:
    cabecalho = pd.read_csv(caminho, nrows=0).columns
    _verificar_colunas("reviews", cabecalho)
    df = pd.read_csv(caminho, usecols=COLUNAS["reviews"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def ler_calendar(caminho: str | Path) -> pd.DataFrame:
    cabecalho = pd.read_csv(caminho, nrows=0).columns
    _verificar_colunas("calendar", cabecalho)
    df = pd.read_csv(caminho, usecols=COLUNAS["calendar"], dtype={"available": "category"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def converter_preco(serie: pd.Series) -> pd.Series:
    """Converte textos como '$1,234.00' para número; valores inválidos viram NaN."""
    texto = serie.astype("string").str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(texto.replace("", pd.NA), errors="coerce").astype(float)


def data_da_coleta(listings: pd.DataFrame) -> pd.Timestamp:
    """Extrai a data da coleta do scrape_id (formato AAAAMMDDhhmmss).

    O Inside Airbnb atribui um único scrape_id por coleta; a data de
    atualização de cada anúncio (last_scraped) pode variar alguns dias.
    """
    ids = pd.Series(listings["scrape_id"].dropna().astype("int64").unique())
    if len(ids) != 1:
        raise ErroDados(
            f"O arquivo listings contém {len(ids)} scrape_id distintos; "
            "esperava-se uma única coleta."
        )
    return pd.to_datetime(str(ids.iloc[0])[:8], format="%Y%m%d")
