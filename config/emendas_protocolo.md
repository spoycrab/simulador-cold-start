# Emendas ao protocolo pré-registrado

Todas as emendas abaixo foram feitas **antes** da execução do experimento principal. Até então,
nenhuma estimativa de efeito das políticas A, C ou D havia sido observada: o comando `piloto`
informa apenas a precisão Monte Carlo e omite as estimativas, justamente para preservar o
pré-registro. Hipóteses (H1 a H4), métricas primárias, margem δ = 0,02, horizonte h = 24 e
comparações primárias (C vs B, D vs B) permanecem inalterados.

## Emenda 1: compatibilidade por consulta

**Motivo.** Sem heterogeneidade entre consultas, todas as buscas de uma faixa de preço viam a mesma
lista; com o volume de contratações observado, a demanda nunca chegava aos novatos (99% sem
avaliação em 24 meses).

**Alteração.** Cada consulta é compatível com cada fornecedor ativo da faixa com probabilidade f
(datas, local e requisitos), sorteada como fluxo exógeno, igual em todas as políticas. f passa a ser
calibrado. A utilidade de não contratar u₀ deixa de ser calibrada e passa a ser hipótese (2,5, com
sensibilidade em 1,5 e 3,5), porque os alvos disponíveis não a separam da demanda.

## Emenda 2: alvo de calibração da compatibilidade

**Motivo.** O tempo de cadastro dos dados mede a idade da *conta* do anfitrião, não a do anúncio
atual. A fração sem nenhuma avaliação é estável entre idades de conta (13% a 27%), mas a fração "sem
avaliação nos primeiros 24 meses" sobe de 20% (contas de 2 anos) para 60% (contas de 10 anos)
(tabela `avaliacao_por_idade_conta` da auditoria).

**Alteração.** O alvo de f passa a ser a fração da coorte de 12 a 23 meses sem nenhuma avaliação
(13,7%), em que as idades da conta e do anúncio praticamente coincidem. As medidas contadas a partir
do cadastro em contas antigas ficam como referência não comparável.

## Emenda 3: política de referência

**Motivo.** Calibrado sob B, o simulador reproduzia o volume e a fração de novatos sem avaliação, mas
os novatos já descobertos recebiam muito menos demanda que nos dados.

**Alteração.** A calibração, a validação e o aquecimento passam a usar a política D como
representação da prática atual (reputação com impulso a novos anúncios, como descrito por Haldar et
al., 2020, para o Airbnb). B passa a ser o contrafactual de reputação pura. As comparações primárias
continuam C vs B e D vs B; a leitura de D vs B passa a ser "o valor do impulso a novatos em relação à
reputação pura".

## Emenda 4: forma do teste de não inferioridade

O critério de H2 é o limite inferior do IC 90% bilateral (bootstrap percentil pareado) de
Δ = NDCG@10(P) − NDCG@10(B) maior que −δ, o que equivale a um teste unilateral com α = 5%.

## Limitação declarada da validação

Calibrado sob D, o simulador reproduz os dois alvos de calibração (volume de avaliações e fração da
coorte de 12 a 23 meses sem avaliação), mas subestima a demanda dos novatos já descobertos: mediana de
7 avaliações na coorte de 12 a 23 meses, contra 18 nos dados; 51% com a primeira avaliação em até
3 meses, contra 70%. O modelo, portanto, exagera a desvantagem dos novatos, o que tende a ampliar o
efeito das políticas de exploração. A análise de sensibilidade em γ (viés de posição) e τ
(sensibilidade da escolha), os dois parâmetros que controlam o peso do ranking, mostra quanto as
conclusões dependem dessa limitação.

---

# Análise adicional após a revisão: política B2

Pedida no parecer do orientador (outubro de 2026), **depois** de observados os resultados do
experimento principal. Não é uma emenda pré-especificada: é uma análise adicional, descritiva,
declarada como tal. Hipóteses, métricas primárias, margem δ e comparações primárias (C vs B,
D vs B) permanecem inalteradas.

**Motivo.** C e B diferem em dois aspectos ao mesmo tempo: o sinal (contratações por exame em C,
estrelas em B) e a exploração (sorteio da distribuição Beta em C). C vs B não isola o efeito da
exploração.

**Alteração.** A política B2 ordena pela média posterior (α₀ + Hᵢ) / (α₀ + β₀ + Eᵢ), com os
mesmos dados e a mesma priori de C, sem sorteio. A diferença C − B é decomposta, por semente, em
(C − B2), efeito da exploração, e (B2 − B), efeito do sinal. B2 não consome números aleatórios,
de modo que os resultados de A, B, C e D não se alteram.
