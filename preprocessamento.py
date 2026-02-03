import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregamento do dataset
df = pd.read_csv("CarPrice_Assignment.csv")
print(df.head())
print(df.dtypes)

print(df.isnull().sum())
categorical_cols = [ 'CarName', 'fueltype', 'aspiration', 'doornumber',
                     'carbody', 'drivewheel', 'enginelocation', 'enginetype',
                     'cylindernumber', 'fuelsystem' ]

for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

print(df.dtypes)


# Correlação com a variavel alvo "price"
correlation = df.corr()['price'].sort_values(ascending=False)
print("Correlação das variáveis com Price:\n", correlation)


# Matriz de correlação (todas as colunas numéricas)
corr_matrix = df.corr(method='spearman')  # ou 'pearson'

# Plotar
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Matriz de Correlação do Dataset")
plt.show()

# Salvar arquivo ajustado
df.to_csv('CarPrice_dataset_ajustado.csv', index=False)

print("Arquivo 'CarPrice_dataset_ajustado.csv' salvo com sucesso!")
