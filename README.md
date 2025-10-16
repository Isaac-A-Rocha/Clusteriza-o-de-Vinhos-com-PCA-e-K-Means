🧠 Clusterização de Vinhos com PCA e K-Means
🎯 Objetivo

Este projeto tem como objetivo agrupar diferentes tipos de vinhos com base em suas características químicas.
Utiliza-se o PCA (Principal Component Analysis) para reduzir a dimensionalidade dos dados e o K-Means para identificar agrupamentos (clusters) de vinhos semelhantes.

📊 Dataset

O dataset utilizado é o Wine Dataset, disponível no Scikit-learn.
Ele contém 178 amostras de vinhos provenientes de 3 produtores diferentes, descritas por 13 variáveis químicas, como teor alcoólico, magnésio, flavonóides, entre outros.

⚙️ Tecnologias e Bibliotecas

Python 3.x

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

🧩 Etapas do Projeto
1. Carregamento e Padronização dos Dados

Os dados são carregados via load_wine() e normalizados com StandardScaler para garantir que todas as variáveis tenham a mesma escala — essencial para o PCA.

2. Redução de Dimensionalidade com PCA

Aplicação do PCA para reduzir as 13 dimensões originais para 2 componentes principais, facilitando a visualização dos dados em um plano 2D.
O PCA também indica a variância explicada, ou seja, quanta informação dos dados originais foi mantida após a redução.

3. Agrupamento com K-Means

O algoritmo K-Means é aplicado sobre as duas componentes principais.
Define-se k = 3, pois sabemos que existem 3 tipos de vinho no dataset.
O resultado é uma atribuição de cada amostra a um cluster.

4. Visualização dos Resultados

Dois gráficos principais são gerados:

Clusters Encontrados pelo K-Means: mostra os agrupamentos descobertos automaticamente.

Classes Reais dos Vinhos: mostra as classes originais, servindo como base para comparação.

Os centróides dos clusters são marcados com “X” vermelhos.

📈 Resultados e Interpretação

O PCA geralmente explica mais de 55% da variância total dos dados com apenas dois componentes.

O K-Means consegue separar bem os três grupos, mostrando que os vinhos possuem diferenças químicas consistentes.

A comparação entre os gráficos evidencia uma correspondência visual forte entre clusters e classes reais.

🧮 Código Simplificado
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wine_data = load_wine()
X = StandardScaler().fit_transform(wine_data.data)
y_true = wine_data.target

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = labels
df['Classe'] = y_true

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8)
plt.title('Clusterização de Vinhos com PCA e K-Means')
plt.show()

print(f"Variância explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")

🚀 Como Executar no Google Colab

Crie um novo notebook no Google Colab
.

Copie e cole o código acima.

Execute todas as células.

Visualize os gráficos de clusterização e a variância explicada.

📚 Conclusão

A combinação de PCA e K-Means é uma abordagem poderosa para explorar e visualizar dados de alta dimensionalidade.
Mesmo sem rótulos, é possível identificar padrões naturais entre os vinhos, demonstrando a eficácia da clusterização não supervisionada.

🧾 Licença

Este projeto é distribuído sob a licença MIT.
Sinta-se livre para usar, estudar e modificar o código.
