"""Leitura e validação do arquivo de parâmetros (config/parametros.yaml)."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_PADRAO = RAIZ_PROJETO / "config" / "parametros.yaml"

UNIDADES_VALIDAS = {"fornecedor"}
ESCALAS_QUALIDADE = {"percentil", "nota"}


def carregar_config(caminho: str | Path | None = None) -> dict:
    """Lê o YAML de parâmetros e valida sua consistência."""
    caminho = Path(caminho) if caminho else CAMINHO_PADRAO
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {caminho}")
    with open(caminho, encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo)
    validar_config(config)
    return config


def validar_config(config: dict) -> None:
    """Interrompe a execução se houver parâmetros inconsistentes.

    As verificações cobrem as regras do protocolo que, se violadas,
    invalidariam a comparação entre políticas.
    """
    erros: list[str] = []
    for grupo in ("dados", "calibracao", "projeto", "protocolo", "sensibilidade"):
        if grupo not in config:
            erros.append(f"Grupo '{grupo}' ausente no arquivo de parâmetros.")
    if erros:
        raise ValueError("\n".join(erros))

    projeto = config["projeto"]
    protocolo = config["protocolo"]
    calibracao = config["calibracao"]

    if projeto["unidade"] not in UNIDADES_VALIDAS:
        erros.append(f"projeto.unidade deve ser um de {sorted(UNIDADES_VALIDAS)}.")
    if projeto["escala_qualidade"] not in ESCALAS_QUALIDADE:
        erros.append(f"projeto.escala_qualidade deve ser um de {sorted(ESCALAS_QUALIDADE)}.")

    T, K, h = projeto["T"], projeto["K"], projeto["horizonte_h"]
    if not (isinstance(T, int) and T > 0):
        erros.append("projeto.T deve ser inteiro positivo.")
    if not (isinstance(K, int) and K > 0):
        erros.append("projeto.K deve ser inteiro positivo.")
    if isinstance(h, int) and isinstance(T, int) and not (0 < h <= T):
        erros.append("projeto.horizonte_h deve estar entre 1 e T.")

    politica_d = projeto["politica_D"]
    n_reservadas = math.ceil(politica_d["rho"] * K)
    posicoes = politica_d["posicoes_reservadas"]
    if len(posicoes) != n_reservadas:
        erros.append(
            f"politica_D: rho={politica_d['rho']} e K={K} exigem {n_reservadas} "
            f"posições reservadas, mas foram informadas {len(posicoes)}."
        )
    if any(not (1 <= p <= K) for p in posicoes) or len(set(posicoes)) != len(posicoes):
        erros.append("politica_D.posicoes_reservadas deve conter posições distintas entre 1 e K.")

    for nome in ("p_review", "churn_mensal", "percentil_alta_qualidade"):
        if not (0 <= projeto[nome] <= 1):
            erros.append(f"projeto.{nome} deve estar entre 0 e 1.")
    if projeto["sigma_ruido"] <= 0:
        erros.append("projeto.sigma_ruido deve ser positivo.")
    if projeto["n_fornecedores_amostra"] <= 0:
        erros.append("projeto.n_fornecedores_amostra deve ser positivo.")
    if protocolo["delta_nao_inferioridade"] <= 0:
        erros.append("protocolo.delta_nao_inferioridade deve ser positivo.")
    maximo = calibracao.get("max_anuncios_por_fornecedor")
    if maximo is not None and not (isinstance(maximo, int) and maximo >= 1):
        erros.append("calibracao.max_anuncios_por_fornecedor deve ser inteiro >= 1 ou null.")
    sim = config.get("simulacao")
    if sim is not None:
        if sim.get("politica_referencia", "D") not in {"A", "B", "C", "D"}:
            erros.append("simulacao.politica_referencia deve ser A, B, C ou D.")
        if sim["tentativa_politica_C"] not in {"exames", "impressoes"}:
            erros.append("simulacao.tentativa_politica_C deve ser 'exames' ou 'impressoes'.")
        if not (0 <= sim["prob_atraso_avaliacao"] <= 1):
            erros.append("simulacao.prob_atraso_avaliacao deve estar entre 0 e 1.")
        if not (0 < sim["percentil_capacidade"] <= 1):
            erros.append("simulacao.percentil_capacidade deve estar entre 0 e 1.")
        if sim["compatibilidade"] is not None and not (0 < sim["compatibilidade"] <= 1):
            erros.append("simulacao.compatibilidade deve estar entre 0 e 1 (ou null).")
        if sim["consultas_por_lote"] < 1 or sim["aquecimento"] < 0:
            erros.append("simulacao.consultas_por_lote e aquecimento devem ser positivos.")
        if any(not (0 < f <= 1) for f in sim["calibracao"]["grade_compatibilidade"]):
            erros.append("simulacao.calibracao.grade_compatibilidade deve ter valores em (0, 1].")
    if calibracao["n_faixas_preco"] < 1:
        erros.append("calibracao.n_faixas_preco deve ser pelo menos 1.")

    if erros:
        raise ValueError("Parâmetros inconsistentes:\n- " + "\n- ".join(erros))
