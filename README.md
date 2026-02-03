# Projeto de Regressão Linear, Ridge e Lasso

Este projeto tem como objetivo aplicar técnicas de **regressão linear** para prever o preço de veículos, seguindo uma sequência estruturada de pré-processamento, modelagem, avaliação e interpretação dos resultados. O trabalho foi desenvolvido com base no dataset Car Price Prediction, disponibilizado no Kaggle. O foco do projeto não é uma análise exploratória aprofundada, mas sim compreender o dataset, preparar os dados corretamente,
aplicar um modelo de regressão linear, avaliar seu desempenho, e visualizar a relação entre a variável mais correlacionada e o preço.

---

## 📁 Estrutura do Projeto

```
├── preprocessamento.py           # Pré-processamento e correlação
├── regressao_linear_simples.py   # Regressão linear simples
├── linear_ridge_lasso.py         # Linear vs Ridge vs Lasso
├── coeficientes.py               # Coeficientes e seleção de atributos
├── regressao_ajustado.csv
└── README.md
```

---
## 📂 Dataset

Fonte: Kaggle

Link: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

Nome do arquivo original: CarPrice_Assignment.csv

O dataset contém informações de diferentes automóveis, incluindo características como:

* tipo de combustível

* tamanho do motor

* potência

* consumo

* dimensões do veículo

* preço (variável alvo)

---

## Bibliotecas utilizadas

Este projeto foi desenvolvido em Python utilizando bibliotecas amplamente empregadas em análise de dados e aprendizado de máquina, conforme descrito abaixo:

---

Pandas: Biblioteca utilizada para carregamento, manipulação e análise de dados tabulares.
Permite ler arquivos CSV, tratar colunas, selecionar variáveis e realizar análises estatísticas básicas.
```python
import pandas as pd
```
Matplotlib: Biblioteca fundamental para criação de gráficos em Python.
Foi utilizada para plotar gráficos de dispersão, retas de regressão e gráficos de importância dos atributos.
```python
import seaborn as pd
```
Seaborn: Biblioteca de visualização estatística baseada no matplotlib.
Facilita a criação de gráficos mais elegantes, como mapas de correlação, boxplots e distribuições.
```python
import matplotlib.pyplot as plt
```
NumPy: Biblioteca fundamental para operações numéricas e matemáticas em Python.
Foi utilizada para cálculos como o RMSE, manipulação de arrays e operações vetoriais.
```python
import numpy as np
```
train_test_split: Função do scikit-learn utilizada para dividir o dataset em conjuntos de treino e teste, garantindo uma avaliação adequada do modelo.
```python
from sklearn.model_selection import train_test_split
```
StandardScaler: Utilizada para padronização dos dados numéricos, fazendo com que todas as variáveis tenham média 0 e desvio padrão 1.
Essa etapa é essencial para modelos sensíveis à escala, como Ridge e Lasso.
```python
from sklearn.preprocessing import StandardScaler
```
LinearRegression: Modelo de Regressão Linear do scikit-learn.
Foi aplicado tanto na regressão linear simples quanto na regressão linear múltipla.
```python
from sklearn.linear_model import LinearRegression
```
Ridge Regression: Modelo de regressão linear com regularização L2, utilizado para reduzir overfitting e controlar a magnitude dos coeficientes.
```python
from sklearn.linear_model import Ridge
```
Lasso Regression: Modelo de regressão linear com regularização L1, capaz de zerar coeficientes, sendo útil para seleção de atributos e análise de importância das variáveis.
```python
from sklearn.linear_model import Lasso
```
cross_val_score: Função utilizada para aplicar validação cruzada (cross-validation), permitindo avaliar o desempenho dos modelos de forma mais robusta.
```python
from sklearn.model_selection import cross_val_score
```
Métricas de Avaliação: Foram utilizadas métricas para avaliar o desempenho dos modelos de regressão:
RMSE (Root Mean Squared Error): mede o erro médio das previsões.
R² (Coeficiente de Determinação): indica o quanto o modelo explica a variabilidade da variável alvo.
```python
from sklearn.metrics import mean_squared_error, r2_score
```

---

## Pré-processamento e Correlação

**Arquivo:** `preprocessamento.py`

Nesta etapa inicial, foi realizado o preparo dos dados para a modelagem:

1. O dataset original foi carregado.
Inicialmente, o dataset bruto foi carregado utilizando a biblioteca Pandas, e foram realizadas inspeções básicas para compreender sua estrutura:
* Visualização das primeiras linhas do dataset (head)
* Verificação dos tipos de dados de cada coluna
* Verificação de valores ausentes (missing values)
Essa etapa permitiu confirmar que o dataset não possui valores nulos, eliminando a necessidade de técnicas de imputação.

2. Tratamento de variáveis categóricas
O dataset contém diversas variáveis categóricas, como tipo de combustível, carroceria, tipo de motor e sistema de combustível. Como modelos de regressão não trabalham diretamente com dados categóricos em formato textual, foi necessário convertê-los para valores numéricos.
As seguintes colunas categóricas foram identificadas:
* CarName
* fueltype
* aspiration
* doornumber
* carbody
* drivewheel
* enginelocation
* enginetype
* cylindernumber
* fuelsystem
Para isso, foi utilizado o método `pd.factorize()`, que transforma cada categoria em um valor inteiro único. Esse método foi escolhido por ser simples e suficiente para esta etapa exploratória e de modelagem inicial.
Após a conversão, todas as colunas do dataset passaram a possuir valores numéricos.

3. Análise de Correlação e matriz de correlação entre as variáveis numéricas e a variável-alvo (`price`).
Com os dados totalmente numéricos, foi realizada uma análise de correlação entre todas as variáveis e a variável alvo price.
Essa análise teve como objetivo:
Identificar quais atributos possuem maior relação com o preço dos veículos
Auxiliar na seleção das variáveis mais relevantes para os modelos de regressão
As correlações foram ordenadas de forma decrescente, permitindo identificar rapidamente as variáveis mais correlacionadas positiva ou negativamente com o preço.

```
Correlação das variáveis com Price:
 price               1.000000
enginesize          0.874145
curbweight          0.835305
horsepower          0.808139
carwidth            0.759325
carlength           0.682920
...                   ...
fuelsystem         -0.122118
drivewheel         -0.577992
citympg            -0.685751
highwaympg         -0.697599
```
Além da correlação individual com a variável alvo, foi construída uma matriz de correlação completa, considerando todas as colunas numéricas do dataset.
Foi utilizado o método de correlação Spearman, por ser mais robusto a relações não lineares e valores extremos.

![Matriz de Correlação](images/matriz_correlação.png)



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
