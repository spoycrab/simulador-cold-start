"""Dados sintéticos no formato do Inside Airbnb, para testes rápidos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coldstart.configuracao import carregar_config

DATA_COLETA = pd.Timestamp("2026-06-14")


def gerar_listings(n_fornecedores: int = 300, semente: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    linhas = []
    id_anuncio = 1000
    for host in range(1, n_fornecedores + 1):
        n_anuncios = int(rng.integers(1, 4))
        anos = int(rng.integers(0, 10))
        meses = int(rng.integers(0, 12))
        sem_aval = rng.random() < 0.2
        for _ in range(n_anuncios):
            id_anuncio += 1
            n_rev = 0 if sem_aval else int(rng.integers(1, 80))
            entrada = DATA_COLETA - pd.to_timedelta((anos * 12 + meses) * 30.4375, unit="D")
            primeira = None if sem_aval else entrada + pd.Timedelta(days=int(rng.integers(0, 400)))
            linhas.append({
                "id": id_anuncio, "host_id": host, "scrape_id": 20260614072545,
                "last_scraped": "2026-06-15", "host_since": np.nan,
                "hosts_time_as_host_years": anos, "hosts_time_as_host_months": meses,
                "price": f"${rng.integers(80, 2000):,}.00",
                "number_of_reviews": n_rev,
                "number_of_reviews_ltm": 0 if sem_aval else int(rng.integers(0, n_rev + 1)),
                "first_review": None if primeira is None else str(primeira.date()),
                "last_review": None if primeira is None else str(DATA_COLETA.date()),
                "review_scores_rating": np.nan if sem_aval else round(float(rng.uniform(4.0, 5.0)), 2),
                "neighbourhood_cleansed": "Pinheiros", "room_type": "Entire home/apt",
            })
    return pd.DataFrame(linhas)


@pytest.fixture
def listings_df() -> pd.DataFrame:
    return gerar_listings()


@pytest.fixture
def pasta_dados(tmp_path, listings_df):
    pasta = tmp_path / "Dados"
    pasta.mkdir()
    listings_df.to_csv(pasta / "listings.csv.gz", index=False)
    reviews = pd.DataFrame({"listing_id": listings_df["id"].repeat(2).to_numpy(),
                            "date": "2025-01-01"})
    reviews.to_csv(pasta / "reviews_csv.gz", index=False)
    datas = pd.date_range(DATA_COLETA, periods=30)
    calendar = pd.DataFrame({
        "listing_id": np.repeat(listings_df["id"].to_numpy(), len(datas)),
        "date": np.tile(datas.strftime("%Y-%m-%d"), len(listings_df)),
        "available": "t",
    })
    calendar.to_csv(pasta / "calendar_csv.gz", index=False)
    return pasta


@pytest.fixture
def config():
    cfg = carregar_config()
    cfg["projeto"]["n_fornecedores_amostra"] = 50
    return cfg
