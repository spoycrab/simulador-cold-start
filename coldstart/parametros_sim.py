"""Parâmetros da simulação, montados a partir do YAML e da calibração (etapa 1).

Cada parâmetro tem uma origem declarada:
  [dados]    estimado diretamente dos dados (etapa 1);
  [calib]    ajustado pelo simulador aos alvos de calibração (etapa 2):
             clientes por rodada e compatibilidade f;
  [projeto]  escolha do autor;
  [hipótese] não identificado pelos dados, com faixa de sensibilidade.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ParametrosSimulacao:
    # População inicial [dados]: um elemento por fornecedor da amostra
    faixa0: np.ndarray
    q0: np.ndarray             # qualidade dos avaliados; NaN = sorteada
    n_aval0: np.ndarray
    soma0: np.ndarray          # soma das notas (escala 1 a 5)
    idade0: np.ndarray         # meses de cadastro na data da coleta
    n_faixas: int
    participacao_oferta: np.ndarray    # [dados] fração de fornecedores por faixa (entrantes)
    participacao_demanda: np.ndarray   # [dados] fração das consultas por faixa
    media_global: float                # [dados] m da média bayesiana
    quantis_nota: np.ndarray           # [dados] função quantil da nota (101 pontos)
    beta_a: float                      # [dados] qualidade na escala da nota
    beta_b: float
    taxa_entrada: float                # [dados] entradas por rodada / N0
    capacidade: int                    # [dados] contratações máximas por rodada

    # Mercado
    escala_qualidade: str = "percentil"   # [projeto]
    aquecimento: int = 24                 # [projeto]
    politica_referencia: str = "D"        # [projeto] prática atual: calibração, validação e aquecimento
    T: int = 60                           # [projeto]
    K: int = 10                           # [projeto]
    clientes_por_rodada: float | None = None  # [calib]
    compatibilidade: float | None = None      # [calib] f: P(fornecedor compatível com a consulta)
    consultas_por_lote: int = 250         # [projeto]
    churn: float = 0.02                   # [hipótese]

    # Escolha e avaliações
    gamma: float = 0.6                    # [hipótese]
    tau: float = 5.0                      # [hipótese]
    u0: float = 2.5                       # [hipótese] utilidade de não contratar
    p_review: float = 0.7                 # [hipótese]
    sigma: float = 0.10                   # [hipótese]
    p_atraso: float = 0.5                 # [projeto]

    # Políticas [projeto]
    C_bayes: float = 5.0
    kappa: float = 2.0
    rho: float = 0.2
    posicoes_reservadas: tuple = (3, 8)
    janela_novidade: int = 12
    escore_sem_aval: float = 1.0
    tentativa_C: str = "exames"

    # Métricas [projeto]
    horizonte_h: int = 24
    percentil_alta: float = 0.75
    janela_descoberta: int = 20

    @property
    def N0(self) -> int:
        return int(len(self.faixa0))

    @property
    def T_total(self) -> int:
        return self.aquecimento + self.T

    @property
    def p_exame(self) -> np.ndarray:
        """Probabilidade de exame por posição: min(1, gamma / log2(1 + k))."""
        k = np.arange(1, self.K + 1)
        return np.minimum(1.0, self.gamma / np.log2(1 + k))

    @property
    def v(self) -> np.ndarray:
        """Peso de exposição por posição (Singh e Joachims): 1 / log2(1 + k)."""
        return 1.0 / np.log2(1 + np.arange(1, self.K + 1))

    @property
    def limiar_alta(self) -> float:
        """Qualidade a partir da qual um fornecedor é de alta qualidade."""
        if self.escala_qualidade == "percentil":
            return self.percentil_alta
        return float(stats.beta.ppf(self.percentil_alta, self.beta_a, self.beta_b))

    def qualidade_de_uniforme(self, u: np.ndarray) -> np.ndarray:
        """Converte uniformes em qualidade latente, conforme a escala."""
        if self.escala_qualidade == "percentil":
            return np.asarray(u, dtype=float)
        return stats.beta.ppf(u, self.beta_a, self.beta_b)

    def nota_de_qualidade(self, x: np.ndarray) -> np.ndarray:
        """Converte qualidade percebida em [0, 1] em nota de 1 a 5."""
        x = np.clip(x, 0.0, 1.0)
        if self.escala_qualidade == "percentil":
            return np.interp(x, np.linspace(0, 1, len(self.quantis_nota)), self.quantis_nota)
        return 1.0 + 4.0 * x

    def com(self, **alteracoes) -> "ParametrosSimulacao":
        return dataclasses.replace(self, **alteracoes)

    def resumo(self) -> dict:
        """Parâmetros escalares, para registro nos resultados."""
        saida = {}
        for campo in dataclasses.fields(self):
            valor = getattr(self, campo.name)
            if isinstance(valor, np.ndarray):
                if valor.size <= 10:
                    saida[campo.name] = [float(x) for x in valor]
                continue
            saida[campo.name] = list(valor) if isinstance(valor, tuple) else valor
        saida["N0"] = self.N0
        return saida


def capacidade_dos_dados(amostra: pd.DataFrame, p_review: float, percentil: float) -> int:
    """Percentil das contratações mensais estimadas (avaliações em 12 meses / 12 / p_review)."""
    maduros = amostra[amostra["meses_cadastro"] >= 12]
    contratacoes_mes = maduros["n_avaliacoes_12m"] / 12 / p_review
    return max(1, int(math.ceil(float(contratacoes_mes.quantile(percentil)))))


def carregar_parametros(config: dict, pasta_resultados: str | Path,
                        exigir_calibracao: bool = True) -> ParametrosSimulacao:
    """Monta os parâmetros a partir do YAML, da etapa 1 e (se houver) da calibração da etapa 2."""
    pasta = Path(pasta_resultados)
    arquivo_cal = pasta / "calibracao" / "calibracao.json"
    arquivo_amostra = pasta / "calibracao" / "fornecedores_amostra.csv"
    if not arquivo_cal.exists() or not arquivo_amostra.exists():
        raise FileNotFoundError(
            "Resultados da etapa 1 não encontrados. Rode antes: python -m coldstart calibrar")
    cal = json.loads(arquivo_cal.read_text(encoding="utf-8"))
    amostra = pd.read_csv(arquivo_amostra).sort_values("host_id").reset_index(drop=True)

    proj, sim = config["projeto"], config["simulacao"]
    escala = proj["escala_qualidade"]
    q0 = (amostra["q_percentil"] if escala == "percentil" else amostra["q_nota"]).to_numpy(float)
    avaliado = amostra["nota_media"].notna().to_numpy()
    n_aval0 = np.where(avaliado, amostra["n_avaliacoes"].to_numpy(), 0).astype(np.int64)
    soma0 = np.where(avaliado, amostra["nota_media"].fillna(0).to_numpy() * n_aval0, 0.0)

    n_faixas = int(cal["populacao"]["n_faixas_preco"])
    oferta = np.bincount(amostra["faixa_preco"], minlength=n_faixas).astype(float)
    demanda = (amostra.groupby("faixa_preco")["n_avaliacoes_12m"].sum()
               .reindex(range(n_faixas), fill_value=0).to_numpy(float))
    beta = cal["qualidade"]["beta_escala_nota"]

    clientes = proj["clientes_por_rodada"]
    compat = sim["compatibilidade"]
    arquivo_sim = pasta / "simulacao" / "calibracao_simulador.json"
    if arquivo_sim.exists():
        calibrado = json.loads(arquivo_sim.read_text(encoding="utf-8"))["escolhido"]
        clientes = calibrado["clientes_por_rodada"] if clientes is None else clientes
        compat = calibrado["compatibilidade"] if compat is None else compat
    if exigir_calibracao and (clientes is None or compat is None):
        raise FileNotFoundError(
            "Demanda e compatibilidade ainda não calibradas. Rode antes: "
            "python -m coldstart calibrar-simulador")

    politica_d = proj["politica_D"]
    return ParametrosSimulacao(
        faixa0=amostra["faixa_preco"].to_numpy(np.int64),
        q0=np.where(avaliado, q0, np.nan),
        n_aval0=n_aval0, soma0=soma0,
        idade0=amostra["meses_cadastro"].to_numpy(np.int64),
        n_faixas=n_faixas,
        participacao_oferta=oferta / oferta.sum(),
        participacao_demanda=demanda / demanda.sum(),
        media_global=float(cal["nota"]["media_global"]),
        quantis_nota=np.asarray(cal["nota"]["quantis"], float),
        beta_a=float(beta["a"]), beta_b=float(beta["b"]),
        taxa_entrada=float(cal["entrada"]["taxa_mensal"]),
        capacidade=capacidade_dos_dados(amostra, proj["p_review"], sim["percentil_capacidade"]),
        escala_qualidade=escala,
        aquecimento=int(sim["aquecimento"]), T=int(proj["T"]), K=int(proj["K"]),
        politica_referencia=str(sim["politica_referencia"]),
        clientes_por_rodada=clientes, compatibilidade=compat,
        consultas_por_lote=int(sim["consultas_por_lote"]),
        churn=float(proj["churn_mensal"]),
        gamma=float(proj["escolha"]["gamma"]), tau=float(proj["escolha"]["tau"]),
        u0=float(proj["escolha"]["u0"]),
        p_review=float(proj["p_review"]), sigma=float(proj["sigma_ruido"]),
        p_atraso=float(sim["prob_atraso_avaliacao"]),
        C_bayes=float(proj["C_bayes"]), kappa=float(proj["kappa_priori"]),
        rho=float(politica_d["rho"]), posicoes_reservadas=tuple(politica_d["posicoes_reservadas"]),
        janela_novidade=int(proj["janela_novidade_w"]),
        escore_sem_aval=float(sim["escore_A_sem_avaliacao"]),
        tentativa_C=sim["tentativa_politica_C"],
        horizonte_h=int(proj["horizonte_h"]),
        percentil_alta=float(proj["percentil_alta_qualidade"]),
        janela_descoberta=int(proj["janela_descoberta"]),
    )
