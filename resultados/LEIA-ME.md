# Resultados do artigo

Esta pasta guarda os resultados usados no artigo (experimento principal com 30 sementes,
análise e sensibilidade), para que as tabelas e figuras possam ser conferidas sem rodar a
simulação. Rodar `./reproduzir.sh` regenera todos os arquivos daqui.

Arquivos principais:

- `experimento/metricas_por_semente.csv`: uma linha por semente e política (base das Tabelas 1 e 2);
- `analise/hipoteses.csv` e `analise/comparacoes.csv`: decisões de H1 a H4 e diferenças pareadas;
- `sensibilidade/sensibilidade.csv` e `sensibilidade/cenarios/`: análise de sensibilidade;
- `simulacao/calibracao_simulador.json` e `simulacao/validacao.csv`: calibração e validação;
- `auditoria/manifesto_dados.json`: nomes, tamanhos e SHA-256 dos arquivos de dados usados.
