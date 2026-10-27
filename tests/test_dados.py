import hashlib

import pandas as pd
import pytest

from coldstart import dados


def test_localiza_arquivos_com_nomes_variados(pasta_dados):
    arquivos = dados.localizar_arquivos(pasta_dados)
    assert arquivos["listings"].name == "listings.csv.gz"
    assert arquivos["reviews"].name == "reviews_csv.gz"
    assert arquivos["calendar"].name == "calendar_csv.gz"


def test_listings_ausente_gera_erro(tmp_path):
    (tmp_path / "Dados").mkdir()
    with pytest.raises(dados.ErroDados):
        dados.localizar_arquivos(tmp_path / "Dados")


def test_dois_listings_gera_erro(pasta_dados, listings_df):
    listings_df.to_csv(pasta_dados / "listings_outra_coleta.csv", index=False)
    with pytest.raises(dados.ErroDados):
        dados.localizar_arquivos(pasta_dados)


def test_escolhe_calendario_da_mesma_coleta(pasta_dados, listings_df):
    outro = pd.DataFrame({"listing_id": listings_df["id"], "date": "2026-09-16", "available": "t"})
    outro.to_csv(pasta_dados / "calendar_setembro.csv.gz", index=False)
    arquivos = dados.localizar_arquivos(pasta_dados)
    assert arquivos["calendar"].name == "calendar_csv.gz"


def test_sha256_igual_ao_hashlib(pasta_dados):
    caminho = pasta_dados / "listings.csv.gz"
    assert dados.sha256_arquivo(caminho) == hashlib.sha256(caminho.read_bytes()).hexdigest()


def test_data_da_coleta_pelo_scrape_id(listings_df):
    assert dados.data_da_coleta(listings_df) == pd.Timestamp("2026-06-14")


def test_converte_preco():
    serie = pd.Series(["$1,234.00", "$80.50", None, ""])
    convertido = dados.converter_preco(serie)
    assert convertido.iloc[0] == 1234.0
    assert convertido.iloc[1] == 80.5
    assert convertido.iloc[2:].isna().all()


def test_coluna_ausente_gera_erro(tmp_path, listings_df):
    caminho = tmp_path / "listings.csv"
    listings_df.drop(columns=["price"]).to_csv(caminho, index=False)
    with pytest.raises(dados.ErroDados, match="price"):
        dados.ler_listings(caminho)
