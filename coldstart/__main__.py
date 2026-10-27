"""Interface de linha de comando.

Uso (na raiz do projeto, com o ambiente virtual ativo):

    python -m coldstart auditar               confere os dados                  (etapa 1)
    python -m coldstart calibrar              estima parâmetros e gera a amostra (etapa 1)
    python -m coldstart calibrar-simulador    ajusta demanda e compatibilidade  (etapa 2)
    python -m coldstart validar               compara simulado e observado      (etapa 2)
    python -m coldstart piloto                precisão Monte Carlo e tempo      (etapa 2)
    python -m coldstart experimento           experimento principal A, B, B2, C, D (etapa 3)
    python -m coldstart analisar              testes, tabelas e figuras         (etapa 3)
    python -m coldstart sensibilidade         um parâmetro por vez              (etapa 3)

Opções comuns:
    --config  caminho do YAML de parâmetros (padrão: config/parametros.yaml)
    --dados   pasta com os arquivos do Inside Airbnb (padrão: valor do YAML)
    --saida   pasta de resultados (padrão: resultados/)
    --processos  núcleos em paralelo (padrão: valor do YAML; 0 = automático)
    --rapido     calibrar-simulador com menos sementes e pontos (apenas para teste)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .configuracao import RAIZ_PROJETO, carregar_config
from .dados import ErroDados


def _argumentos(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m coldstart",
                                     description="Calibração e simulação do cold start de prestadores.")
    parser.add_argument("comando", choices=["auditar", "calibrar", "calibrar-simulador",
                                            "validar", "piloto", "experimento", "analisar",
                                            "sensibilidade"])
    parser.add_argument("--sementes", type=int, default=None,
                        help="número de sementes (experimento e sensibilidade)")
    parser.add_argument("--config", default=None, help="arquivo YAML de parâmetros")
    parser.add_argument("--dados", default=None, help="pasta com listings, reviews e calendar")
    parser.add_argument("--saida", default=None, help="pasta de resultados")
    parser.add_argument("--processos", type=int, default=None, help="núcleos em paralelo")
    parser.add_argument("--rapido", action="store_true",
                        help="calibrar-simulador com menos sementes e pontos (só para teste)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _argumentos(argv)
    config = carregar_config(args.config)
    pasta_dados = Path(args.dados or RAIZ_PROJETO / config["dados"]["pasta"])
    pasta_saida = Path(args.saida or RAIZ_PROJETO / "resultados")
    inicio = time.perf_counter()
    try:
        if args.comando == "auditar":
            from .auditoria import executar_auditoria
            tabela = executar_auditoria(config, pasta_dados, pasta_saida / "auditoria")
            print(tabela.to_string(index=False))
            print(f"\nResultados em {pasta_saida / 'auditoria'}")
        elif args.comando == "calibrar":
            from .calibracao import executar_calibracao
            res = executar_calibracao(config, pasta_dados, pasta_saida / "calibracao")
            _resumo_calibracao(res)
            print(f"\nResultados em {pasta_saida / 'calibracao'}")
        else:
            from . import calibracao_sim
            from .experimento import numero_de_processos
            processos = numero_de_processos(
                args.processos if args.processos is not None else config["simulacao"]["processos"])
            print(f"Usando {processos} processo(s).")
            if args.comando == "calibrar-simulador":
                res = calibracao_sim.calibrar_simulador(config, pasta_saida, processos, args.rapido)
                e = res["escolhido"]
                print(f"\nCompatibilidade f = {e['compatibilidade']:.4f}; "
                      f"clientes por rodada = {e['clientes_por_rodada']:.0f}")
                chave = res["alvo_compatibilidade"]
                print(f"Volume: {e['avaliacoes_por_fornecedor_mes']:.3f} "
                      f"(alvo {res['alvos']['avaliacoes_por_fornecedor_mes']:.3f}); "
                      f"{chave}: {e[chave]:.3f} (alvo {res['alvos'][chave]:.3f})")
            elif args.comando == "validar":
                tabela = calibracao_sim.validar(config, pasta_saida, processos)
                print(tabela.round(3).to_string(index=False))
            elif args.comando == "experimento":
                from .experimento import executar_experimento
                df = executar_experimento(config, pasta_saida, args.sementes, processos)
                print(f"{df['semente'].nunique()} sementes concluídas. "
                      "Rode agora: python -m coldstart analisar")
                print(f"\nResultados em {pasta_saida / 'experimento'}")
                print(f"Concluído em {time.perf_counter() - inicio:.1f} s.")
                return 0
            elif args.comando == "analisar":
                from .analise import executar_analise
                res = executar_analise(config, pasta_saida)
                colunas = ["hipotese", "comparacao", "metrica", "diferenca_media", "sustentada"]
                print(res["hipoteses"][colunas].round(4).to_string(index=False))
                decomposicao = res.get("decomposicao")
                if decomposicao is not None and not decomposicao.empty:
                    print("\nDecomposição C - B = (C - B2) + (B2 - B) (análise adicional):")
                    print(decomposicao[["metrica", "parcela", "diferenca_media", "ic95_inf",
                                        "ic95_sup", "fracao_do_total"]].round(4).to_string(index=False))
                print(f"\nResultados em {pasta_saida / 'analise'}")
                print(f"Concluído em {time.perf_counter() - inicio:.1f} s.")
                return 0
            elif args.comando == "sensibilidade":
                from .sensibilidade import executar_sensibilidade
                tabela = executar_sensibilidade(config, pasta_saida, args.sementes, processos)
                colunas = ["cenario", "comparacao", "delta_rmst_alta", "H1_sustentada",
                           "delta_ndcg10", "H2_sustentada"]
                print(tabela[colunas].round(4).to_string(index=False))
                print(f"\nResultados em {pasta_saida / 'sensibilidade'}")
                print(f"Concluído em {time.perf_counter() - inicio:.1f} s.")
                return 0
            elif args.comando == "piloto":
                res = calibracao_sim.piloto(config, pasta_saida, processos)
                print(pd.DataFrame(res["precisao"]).round(4).to_string(index=False))
                print(f"\nSementes recomendadas: {res['sementes_recomendadas']}; "
                      f"tempo estimado do experimento: {res['estimativa_minutos_experimento']} min "
                      f"com {processos} processo(s).")
                print(res["observacao"])
            print(f"\nResultados em {pasta_saida / 'simulacao'}")
    except ErroDados as erro:
        print(f"ERRO NOS DADOS: {erro}", file=sys.stderr)
        return 1
    except FileNotFoundError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1
    print(f"Concluído em {time.perf_counter() - inicio:.1f} s.")
    return 0


def _resumo_calibracao(res: dict) -> None:
    pop = res["populacao"]
    print(f"Data da coleta: {res['data_coleta']}")
    print(f"Anfitriões na coleta: {pop['fornecedores_total']}; elegíveis "
          f"(até {pop['max_anuncios_por_fornecedor']} anúncio(s)): {pop['fornecedores_elegiveis']}")
    print(f"Fornecedores com preço: {pop['fornecedores_com_preco']} "
          f"(excluídos sem preço: {pop['excluidos_sem_preco']})")
    print(f"Fornecedores por faixa de preço: {pop['fornecedores_por_faixa']}")
    beta = res["qualidade"]["beta_escala_nota"]
    print(f"Qualidade (escala da nota): Beta(a={beta['a']:.3f}, b={beta['b']:.3f})")
    print(f"Taxa mensal de entrada ({res['entrada']['ano_referencia']}): "
          f"{100 * res['entrada']['taxa_mensal']:.2f}%")
    print("Alvos de calibração:")
    for nome, item in res["alvos_calibracao"].items():
        print(f"  {nome}: {item['valor']:.4f}")
    print("Alvos de validação:")
    for nome, item in res["alvos_validacao"].items():
        print(f"  {nome}: {item['valor']:.4f}")
    print(f"Amostra: {res['amostra']['n']} fornecedores, por faixa {res['amostra']['por_faixa']}")


if __name__ == "__main__":
    sys.exit(main())
