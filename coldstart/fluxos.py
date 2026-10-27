"""Fluxos exógenos: números aleatórios comuns entre as políticas.

Tudo o que não depende da política é sorteado aqui, a partir da semente:
entradas, faixa e qualidade dos entrantes, saídas, chaves de desempate,
número de consultas, sorteios de exame e de escolha, ocorrência, ruído e
atraso das avaliações.

* Eventos por fornecedor (saída, avaliações) ficam em matrizes indexadas
  pelo fornecedor e pela rodada ou pelo número da contratação. A n-ésima
  contratação de um fornecedor usa sempre os mesmos sorteios, qualquer que
  seja a política.
* Sorteios por consulta (exame e escolha) vêm de um gerador identificado
  por (semente, rodada, faixa, lote). A ordem em que cada política consome
  números aleatórios não altera esses sorteios.

Com isso, a n-ésima semente de cada política enfrenta exatamente o mesmo
mercado, e as diferenças entre políticas não incluem ruído de sorteio.
"""

from __future__ import annotations

import numpy as np

from .parametros_sim import ParametrosSimulacao

FLUXO_BASE, FLUXO_LOTE, FLUXO_THOMPSON = 0, 1, 2


class Fluxos:
    def __init__(self, semente: int, p: ParametrosSimulacao):
        if p.clientes_por_rodada is None:
            raise ValueError("clientes_por_rodada não definido.")
        self.semente = int(semente)
        rng = np.random.default_rng([self.semente, FLUXO_BASE])
        N0, TT = p.N0, p.T_total

        # Entradas: Poisson por rodada, proporcional ao tamanho inicial do mercado
        self.n_entradas = rng.poisson(p.taxa_entrada * N0, size=TT).astype(np.int64)
        self.inicio_entrada = N0 + np.concatenate([[0], np.cumsum(self.n_entradas)[:-1]])
        self.N_max = N0 + int(self.n_entradas.sum())
        n_entrantes = self.N_max - N0
        self.faixa_entrante = rng.choice(p.n_faixas, size=n_entrantes, p=p.participacao_oferta)

        # Atributos por fornecedor (incumbentes e entrantes)
        self.u_qualidade = rng.random(self.N_max)    # qualidade de quem não tem nota observada
        self.chave = rng.random(self.N_max)          # desempate fixo entre escores iguais
        self.u_saida = rng.random((self.N_max, TT), dtype=np.float32)

        # Sorteios da n-ésima contratação de cada fornecedor
        maximo = p.capacidade * TT
        self.u_avaliacao = rng.random((self.N_max, maximo), dtype=np.float32)
        self.z_avaliacao = rng.standard_normal((self.N_max, maximo), dtype=np.float32)
        self.u_atraso = rng.random((self.N_max, maximo), dtype=np.float32)

        # Consultas por rodada e faixa (determinísticas, para uma calibração suave)
        por_faixa = np.round(p.clientes_por_rodada * p.participacao_demanda).astype(np.int64)
        self.consultas = np.tile(por_faixa, (TT, 1))

    def gerador_lote(self, t: int, faixa: int, lote: int) -> np.random.Generator:
        """Sorteios de exame e de escolha de um lote de consultas."""
        return np.random.default_rng([self.semente, FLUXO_LOTE, t, faixa, lote])

    def gerador_thompson(self, t: int, faixa: int, lote: int) -> np.random.Generator:
        """Amostras da política C (próprias da política, mas reproduzíveis)."""
        return np.random.default_rng([self.semente, FLUXO_THOMPSON, t, faixa, lote])
