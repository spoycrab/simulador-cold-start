# Simulador do cold start de prestadores em marketplaces de serviços

Código do Trabalho de Conclusão de Curso **"Mitigação do Cold Start de Prestadores em Marketplaces
de Serviços por Exploração Controlada"**, de Lucas Laranja Alcantara (Sistemas de Informação,
Universidade Vila Velha, 2026).

O simulador é baseado em agentes e calibrado com dados públicos do Inside Airbnb de São Paulo. Ele
compara quatro políticas de ranqueamento quanto ao tempo até a primeira contratação de novos
prestadores e à qualidade da lista exibida ao cliente:

- **A**: média simples das avaliações;
- **B**: reputação bayesiana;
- **C**: Thompson Sampling (Beta-Bernoulli, contratação por exame);
- **D**: cota temporal de exposição para novatos (reputação bayesiana com posições reservadas).

O código está organizado em três etapas: (1) auditoria dos dados e estimação dos parâmetros
identificáveis; (2) simulador, calibração e validação; (3) experimento principal com 30 sementes
pareadas, testes das hipóteses pré-especificadas e análise de sensibilidade.

## Reprodução rápida

```bash
pip install -r requirements.txt
./reproduzir.sh             # todas as etapas; requer os dados em Dados/
```

Os dados do Inside Airbnb não estão no repositório (veja `Dados/LEIA-ME.txt`).
## Licença

Código sob a licença MIT (arquivo `LICENSE`). Os dados do Inside Airbnb são distribuídos pelo
próprio projeto sob a licença CC BY 4.0.

## Instalação

Requer Python 3.10 ou superior.

```bash
cd simulador-cold-start
pip install -r requirements.txt
```

## Dados

Coloque na pasta `Dados/` os três arquivos da **mesma coleta** (14/06/2026):

| Arquivo | Obrigatório | Uso |
|:---|:---:|:---|
| `listings.csv.gz` | sim | anúncios e anfitriões |
| `reviews.csv.gz` | não | conferência de `first_review` e `number_of_reviews` |
| `calendar.csv.gz` | não | conferência de cobertura e indisponibilidade |

Os nomes são reconhecidos pelo prefixo (`listings`, `reviews`, `calendar`), com extensão `.csv` ou
`.csv.gz`. Se houver mais de um calendário na pasta, o programa escolhe automaticamente o que começa
na data da coleta.

## Execução

```bash
# Etapa 1
python -m coldstart auditar              # confere os dados (~30 s)
python -m coldstart calibrar             # estima parâmetros e gera a amostra (~10 s)

# Etapa 2 (nesta ordem; rode de novo após mudar politica_referencia)
python -m coldstart calibrar-simulador   # ajusta demanda e compatibilidade (~4 min com 2 núcleos)
python -m coldstart validar              # compara simulado e observado (~15 s)
python -m coldstart piloto               # precisão Monte Carlo e tempo estimado (~2 a 4 min)

# Etapa 3
python -m coldstart experimento          # 30 sementes, A, B, C, D (~2 min com 11 núcleos)
python -m coldstart analisar             # testes, tabelas e figuras (segundos)
python -m coldstart sensibilidade        # ~20 cenários x 10 sementes (~15 min com 11 núcleos)

pytest                                   # 47 testes com dados sintéticos (~6 s)
```

Os tempos foram medidos com 2 núcleos; com mais núcleos, as etapas 2 ficam proporcionalmente mais
rápidas. Por padrão o programa usa todos os núcleos menos um (`simulacao.processos` no YAML, ou
`--processos N`). Para um teste rápido da calibração: `python -m coldstart calibrar-simulador --rapido`
(resultado grosseiro, só para conferir que o programa roda).

Opções: `--config caminho.yaml`, `--dados pasta`, `--saida pasta`, `--processos N`, `--rapido`.

## Saídas

`resultados/auditoria/`

- `auditoria.csv` e `.tex`: todas as estatísticas e verificações, incluindo as que **não** reproduzem
  o texto original (por exemplo, o 99,5% sobre a data de cadastro);
- `entradas_por_ano.csv` e `.tex`: fornecedores ativos por ano de entrada reconstruído;
- `manifesto_dados.json`: nome, tamanho e SHA-256 de cada arquivo usado.

`resultados/calibracao/`

- `calibracao.json`: tudo o que a etapa 2 vai ler (faixas de preço, quantis da nota, distribuição da
  qualidade, taxa de entrada, alvos de calibração e de validação, hipóteses não identificadas);
- `fornecedores_amostra.csv`: a amostra estratificada usada na simulação;
- `parametros_calibrados`, `alvos` e `comparacao_amostra` em `.csv` e `.tex`.

`resultados/simulacao/`

- `calibracao_simulador.json`: demanda e compatibilidade escolhidas, todos os pontos avaliados e a
  curva de identificação;
- `identificacao.csv` e `.tex`: para cada compatibilidade f da grade, a demanda que reproduz o
  volume e a fração sem avaliação resultante;
- `validacao.csv` e `.tex`: momentos observados e simulados, separados em calibração, validação e
  referência não comparável;
- `piloto.json`: meia largura dos intervalos das comparações primárias e número de sementes
  recomendado. As estimativas de efeito não são exibidas, para preservar o pré-registro.

`resultados/experimento/`: `metricas_por_semente.csv` (uma linha por semente e política),
`km_alta.csv` (curvas de Kaplan–Meier por semente) e `parametros.json` (parâmetros usados).

`resultados/analise/`

- `hipoteses`: decisão de H1 a H4 para C vs B e D vs B, com o critério pré-registrado;
- `comparacoes`: diferenças pareadas (P − B) de todas as métricas, IC 95% e 90% por bootstrap,
  valor-p de Wilcoxon, Holm, tamanho de efeito d_z e erro Monte Carlo;
- `metricas_por_politica`: média e IC 95% de cada métrica por política;
- `nao_inferioridade_por_delta`: decisão de H2 com δ = 0,01, 0,02 e 0,05;
- `fig_kaplan_meier_alta.pdf`: proporção de novatos de alta qualidade ainda não contratados;
- `fig_fronteira.pdf`: Δ NDCG@10 (cliente) contra Δ RMST (novato), em relação a B.

`resultados/sensibilidade/`: um CSV por cenário em `cenarios/`, a tabela `sensibilidade` e a
figura `fig_sensibilidade.pdf`. Se o comando for interrompido, os cenários concluídos são
reaproveitados.

As tabelas `.tex` usam vírgula decimal e podem ser coladas no artigo (modelo SBC).

## Decisões de modelagem

Todas estão em `config/parametros.yaml` e `config/emendas_protocolo.md`, separadas em *dados*, *calibração*, *projeto* (escolhas do
autor), *protocolo* (pré-registro) e *sensibilidade*.

| Decisão | Valor | Justificativa resumida |
|:---|:---|:---|
| Unidade ranqueada | fornecedor (anfitrião) | entrada só pode ser calibrada por anfitrião |
| População | anfitriões com **um único anúncio** | fornecedor e item ranqueado coincidem; anfitriões com muitos anúncios operam como empresas e distorcem a demanda |
| Categorias de consulta (G) | 5 quintis do preço | categorias equilibradas por construção |
| Amostra | 2.000, estratificada por faixa de preço e ausência de nota | preserva a proporção de fornecedores sem histórico |
| Taxa de entrada | taxa de 2025 entre sobreviventes | média histórica mistura períodos diferentes |
| Margem δ | 0,02 de NDCG@10 | pré-registrada; 0,01 e 0,05 na sensibilidade |
| Saída (churn) | 2% ao mês (hipótese) | sensibilidade de 0% a 4% |
| Política D | posições 3 e 8; novato = menos de 12 rodadas | evita o topo da lista |
| Horizonte do TTFC | 24 rodadas | duas vezes a janela de novidade |

### Qualidade latente

As notas estão comprimidas no topo (80% dos fornecedores com 4,8 ou mais). Se a qualidade latente for
a nota em escala [0, 1], a distribuição ajustada é muito concentrada perto de 1 e o NDCG perde
capacidade de distinguir rankings. Por isso há duas escalas, escolhidas em `projeto.escala_qualidade`:

- `percentil` (padrão): a qualidade é o percentil da nota entre os avaliados, uniforme em (0, 1). A
  ordem das notas é preservada e as notas simuladas são geradas pela função quantil observada
  (`nota.quantis` em `calibracao.json`).
- `nota`: a qualidade segue uma Beta ajustada às notas, descontado o ruído das avaliações.

### O que os dados não identificam

γ (viés de posição), τ (sensibilidade da escolha), p_review, σ (ruído das avaliações) e a taxa de
saída são hipóteses declaradas, com grades de sensibilidade. A opção externa u₀ e o número de
clientes por rodada serão ajustados na etapa 2 para reproduzir os alvos de calibração; os alvos de
validação ficam fora do ajuste.

### Limitações registradas

- A base observa apenas fornecedores ativos na coleta (viés de sobrevivência), inclusive na taxa de
  entrada.
- A data de entrada é reconstruída a partir do tempo de cadastro e tem precisão mensal; em cerca de
  10% dos anúncios a primeira avaliação aparece antes da data reconstruída.
- A primeira avaliação é uma proxy imperfeita da primeira contratação.
- O calendário marca dias indisponíveis sem distinguir reservas de bloqueios do anfitrião.

## Etapa 2: o simulador

Cada semente executa 24 rodadas de aquecimento sob a **política de referência**, a partir da amostra
real. O estado
ao fim do aquecimento é copiado, e A, B, C e D rodam as 60 rodadas seguintes com os mesmos fluxos
exógenos (números aleatórios comuns). A cada rodada (um mês):

1. fornecedores saem com probabilidade `churn_mensal` e entrantes chegam sem histórico;
2. as consultas de cada faixa de preço chegam em lotes, em número proporcional às avaliações
   observadas na faixa;
3. cada consulta é compatível com cada fornecedor da faixa com probabilidade f (datas, local e
   requisitos); a política ordena os compatíveis com capacidade e exibe K = 10;
4. a posição k é examinada com probabilidade min(1, γ/log₂(1+k));
5. o cliente escolhe pelo modelo logit entre os examinados e a opção de não contratar (u₀);
6. cada fornecedor atende no máximo a capacidade mensal (percentil 95 dos dados);
7. cada contratação gera avaliação com probabilidade p_review, visível na mesma rodada ou na
   seguinte.

Políticas: A (média simples), B (média bayesiana), C (Thompson Sampling Beta-Bernoulli, com
contratação por exame), D (B com as posições 3 e 8 reservadas a novatos com menos de 12 rodadas, sem
rebaixar novatos que B já colocaria acima).

### Política de referência (prática atual)

A calibração, a validação e o aquecimento usam a política que representa a prática atual da
plataforma (`simulacao.politica_referencia`). O padrão é **D**: reputação com impulso a novos
anúncios, como o descrito por Haldar et al. (2020) para o Airbnb. Com isso, **B passa a ser o
contrafactual de reputação pura**, e não o retrato do status quo.

Motivo: calibrado sob B, o simulador acertava a fração de novatos sem avaliação, mas os novatos
descobertos recebiam pouca demanda (mediana de 5 avaliações em 12 a 23 meses, contra 18 nos dados).
Sob B, o novato entra com a nota média global, abaixo de cerca de 65% dos incumbentes avaliados; nos
dados, ele recebe demanda próxima à média do mercado desde o início. Para comparar, basta trocar
`politica_referencia` para `B` no YAML e rodar de novo a calibração e a validação.

### Calibração da etapa 2

Sob a política de referência:

| Parâmetro | Alvo | Por quê |
|:---|:---|:---|
| Clientes por rodada | 1,09 avaliação por fornecedor por mês | define o volume |
| Compatibilidade f | 13,7% da coorte de 12 a 23 meses sem nenhuma avaliação | define quão fundo a demanda chega na lista |

u₀ é hipótese (2,5, ou seja, a utilidade de um fornecedor mediano), com sensibilidade em 1,5 e 3,5:
com esses alvos, ele não se separa da demanda, porque os dois só mudam o volume.

**Por que a coorte de 12 a 23 meses.** O tempo de cadastro mede a idade da *conta* do anfitrião, não
a do anúncio atual. A tabela `avaliacao_por_idade_conta` mostra que a fração sem nenhuma avaliação é
estável (13% a 27%) em qualquer idade de conta, mas a fração "sem avaliação nos primeiros 24 meses"
sobe de 20% (contas de 2 anos) para 60% (10 anos). Em contas antigas, o anúncio atual foi criado
depois da conta. Por isso, medidas contadas a partir do cadastro só são usadas na coorte jovem; as
demais aparecem na validação como referência não comparável.

## Etapa 3: análise pré-registrada

| Hipótese | Métrica | Critério |
|:---|:---|:---|
| H1 | RMST do TTFC, novatos de alta qualidade (h = 24) | Wilcoxon unilateral (P < B), p de Holm < 0,05 |
| H2 | NDCG@10 | limite inferior do IC 90% de Δ > −δ (δ = 0,02) |
| H3 | Descoberta de novatos de alta qualidade (20 rodadas) | Wilcoxon unilateral (P > B), p de Holm < 0,05 |
| H4 | Gini da exposição normalizada por consultas elegíveis | Wilcoxon unilateral (P < B), p de Holm < 0,05 |

Comparações primárias: C vs B e D vs B (Holm sobre as duas em cada hipótese). A vs B é reportada
como secundária. A sensibilidade varia um parâmetro por vez, mantendo a demanda e a compatibilidade
calibradas no cenário central.

As emendas ao protocolo e a limitação da validação estão em `config/emendas_protocolo.md`. Em
resumo: o simulador reproduz o volume de avaliações e a fração de novatos sem avaliação, mas
subestima a demanda dos novatos já descobertos (mediana de 7 avaliações na coorte de 12 a 23 meses,
contra 18 nos dados). Isso tende a exagerar a desvantagem dos novatos e, portanto, o efeito das
políticas de exploração; os cenários de γ e τ na sensibilidade mostram quanto as conclusões dependem
disso.

## Estrutura

```
coldstart/
  configuracao.py   leitura e validação do YAML
  dados.py          localização, SHA-256 e leitura dos arquivos
  fornecedores.py   agregação por anfitrião e reconstrução da entrada
  auditoria.py      estatísticas descritivas e verificações de consistência
  calibracao.py     faixas de preço, qualidade, entrada, alvos e amostra
  saidas.py         gravação de CSV e LaTeX
  parametros_sim.py parâmetros da simulação e sua origem ([dados], [calib], [projeto], [hipótese])
  fluxos.py         fluxos exógenos com números aleatórios comuns entre políticas
  politicas.py      políticas A, B, C e D
  motor.py          estado do mercado e motor de rodadas
  metricas.py       NDCG, Kaplan–Meier, RMST, Gini, descoberta e momentos simulados
  experimento.py    aquecimento, ramificação pareada e execução em paralelo
  calibracao_sim.py calibração da demanda e da compatibilidade, validação e piloto
  estatistica.py    bootstrap pareado, Wilcoxon, Holm, resumo das diferenças
  analise.py        hipóteses, tabelas e figuras do experimento principal
  sensibilidade.py  cenários um parâmetro por vez, tabela e figura
  __main__.py       linha de comando
config/parametros.yaml
config/emendas_protocolo.md
tests/              testes com dados sintéticos
```
