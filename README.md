# Projeto de Regressão Linear, Ridge e Lasso

Este projeto tem como objetivo aplicar técnicas de **regressão linear** para prever o preço de veículos, seguindo uma sequência estruturada de pré-processamento, modelagem, avaliação e interpretação dos resultados. O trabalho foi desenvolvido com base no dataset Car Price Prediction, disponibilizado no Kaggle. O foco do projeto não é uma análise exploratória aprofundada, mas sim compreender o dataset, preparar os dados corretamente,
aplicar um modelo de regressão linear, avaliar seu desempenho, e visualizar a relação entre a variável mais correlacionada e o preço.

---

## 📁 Estrutura do Projeto

```
├── preprocessamento.py   # Pré-processamento e correlação
├── regressao_linear_simples.py   # Regressão linear simples
├── linear_ridge_lasso.py   # Linear vs Ridge vs Lasso
├── coeficientes.py   # Coeficientes e seleção de atributos
├── regressao_ajustado.csv
└── README.md
```

---
## 📂 Dataset

Fonte: Kaggle

Link: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

Nome do arquivo original: CarPrice_Assignment.csv

O dataset contém informações de diferentes automóveis, incluindo características como:

tipo de combustível,

tamanho do motor,

potência,

consumo,

dimensões do veículo,

e o preço (variável alvo).

---

## Questão 1 – Pré-processamento e Correlação

**Arquivo:** `regressao_q1.py`

Nesta etapa inicial, foi realizado o preparo dos dados para a modelagem:

1. O dataset original foi carregado.
2. Foram tratados valores ausentes, garantindo consistência dos dados.
3. Foi calculada a matriz de correlação entre as variáveis numéricas e a variável-alvo (`price`).
4. Com base na correlação, foi possível identificar quais atributos possuem maior relação com o preço.
5. Os dados numéricos foram padronizados utilizando o `StandardScaler`, garantindo média zero e desvio padrão igual a um.
6. O dataset final pré-processado foi salvo no arquivo `regressao_ajustado.csv`.

Essa etapa é fundamental para garantir qualidade dos dados e evitar vieses nos modelos.

---

## Questão 2 – Regressão Linear Simples

**Arquivo:** `regressao_q2.py`

Nesta etapa, foi aplicada a regressão linear simples com o objetivo de entender a relação entre uma variável explicativa e o preço:

1. Foi utilizado o modelo de **Regressão Linear** da biblioteca `scikit-learn`.
2. A variável mais correlacionada com o preço foi escolhida para o modelo.
3. Um gráfico foi gerado exibindo os pontos reais e a reta de regressão ajustada.
4. O desempenho do modelo foi avaliado utilizando as métricas **RMSE** e **R²**.
5. Os resultados foram analisados, permitindo interpretar o poder explicativo do modelo simples.

Essa análise fornece uma visualização clara da relação linear entre a variável escolhida e o preço.

---

## Questão 3 – Comparação: Linear vs Ridge vs Lasso

**Arquivo:** `regressao_q3.py`

Nesta fase, o objetivo foi comparar diferentes modelos de regressão:

1. Foram aplicados três modelos: **Regressão Linear**, **Ridge** e **Lasso**.
2. Foi utilizada **validação cruzada com 5 folds** (`cross_val_score`) para obter métricas mais robustas.
3. As métricas **RMSE médio** e **R² médio** foram calculadas para cada modelo.
4. Os resultados foram organizados em uma tabela comparativa.
5. O melhor modelo foi identificado com base no menor RMSE.

Essa comparação evidencia o impacto da regularização no desempenho dos modelos.

---

## Questão 4 – Coeficientes e Seleção de Atributos

**Arquivo:** `regressao_q4.py`

Por fim, foi realizada a análise de importância das variáveis utilizando o modelo Lasso:

1. Os dados foram padronizados antes do treinamento do modelo Lasso.
2. Os coeficientes aprendidos pelo modelo foram extraídos.
3. Foi calculada a importância absoluta de cada atributo.
4. Um gráfico de barras horizontais foi gerado para visualizar a importância dos atributos.
5. Os resultados foram discutidos, destacando quais variáveis têm maior impacto na previsão do preço.

O Lasso mostrou-se eficiente para seleção automática de atributos, reduzindo a influência de variáveis menos relevantes.

---

## 📊 Conclusão

O projeto demonstrou, de forma prática, todo o fluxo de uma análise de regressão:

* Pré-processamento adequado dos dados;
* Aplicação de regressão linear simples e múltipla;
* Comparação entre modelos com e sem regularização;
* Interpretação dos coeficientes e seleção de atributos relevantes.

Os resultados obtidos indicam que modelos regularizados, como o Ridge e o Lasso, podem melhorar a generalização e fornecer insights importantes sobre a relevância das variáveis.

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

Projeto desenvolvido para fins acadêmicos e aprendizado em Machine Learning.
