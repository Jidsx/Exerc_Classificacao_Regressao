# Modelos de Dados Supervisionado: Classificação e Regressão

Conjuntos de dados supervisionados são compostos por exemplos de entrada (amostras) associados a rótulos ou resultados conhecidos. Esses rótulos permitem que algoritmos de aprendizado de máquina aprendam a fazer previsões ou classificações precisas quando se deparam com novos dados. Utilizamos um conjunto de treinamento para ensinar os modelos a mostrar o resultado desejado, incluindo entradas (características) e saídas corretas (rótulos). O algoritmo mede sua precisão através de uma função de perda, ajustando-se até que o erro seja minimizado.

Os modelos de Classificação e Regressão aprendem com dados rotulados e, em seguida, podem classificar ou prever informações.

## Modelo de Classificação

Técnicas de classificação mais comuns:

* Árvores de Decisão;
* Regras de Classificação;
* Naive Bayes;
* K-Nearest Neighbors (KNN);
* Máquinas de Vetores de Suporte (SVM).

A classificação é usada para prever a classe ou categoria de um objeto com base em suas características. Vamos considerar um conjunto de dados de e-mails rotulados como “spam” ou “não spam”:

### Naive Bayes
O classificador Naive Bayes é um modelo probabilístico simples e poderoso baseado no Teorema de Bayes com a suposição de independência condicional entre as características. Ele é frequentemente utilizado em problemas de classificação, especialmente em tarefas envolvendo processamento de linguagem natural (NLP) e análise de texto.

~~~python
from sklearn.naive_bayes import GaussianNB

# Dados de Exemplo
x = [[100, 20],[150,30],[120, 25], [140, 28]]
y = ['Não spam', 'spam', 'Não spam', 'spam']

# Treinando o modelo
model = GaussianNB()
model.fit(x,y)

# Previsão para um novo e-mail
novo_email = [[130, 22]]
resultado = model.predict(novo_email)
print(f"Previsão para o novo e-mail: {resultado[0]}")
~~~

![cla_4](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/e85c2949-7b7b-4459-8090-6e8821ae4df3)

~~~python
# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = MultinomialNB() # Criar o modelo Naive Bayes multinomial
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("accuracy: ", accuracy)
~~~

![cla_3](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/7512e6a8-aac5-4ad5-b83a-279264afe2f1)

### K-Nearest Neighbors (KNN)
O algoritmo K-Nearest Neighbors (KNN) é um método de aprendizado de máquina utilizado para classificação e regressão. Ele classifica novos pontos de dados com base na classe predominante entre seus k vizinhos mais próximos no espaço das características.

~~~python
# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = KNeighborsClassifier(n_neighbors=3) # Criar o modelo KNN com 3 vizinhos
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("accuracy: ", accuracy)
~~~

![cla_3](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/7512e6a8-aac5-4ad5-b83a-279264afe2f1)

### Support Vector Machine (SVM)
O Support Vector Machine (SVM), ou Máquina de Vetores de Suporte, é um poderoso algoritmo de aprendizado supervisionado utilizado para classificação, regressão e detecção de outliers. Ele encontra o hiperplano de separação que melhor divide os dados em classes distintas no espaço de características.

~~~python
# Passo 1: Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = SVC(kernel='linear') # Criar o modelo SVM com kernel linear
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("accuracy: ", accuracy)
~~~

![cla_3](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/7512e6a8-aac5-4ad5-b83a-279264afe2f1)

### Árvores de Decisão
A Árvore de Decisão é um modelo de aprendizado de máquina que utiliza uma estrutura em forma de árvore para tomar decisões com base nas características dos dados. É uma técnica popular em problemas de classificação e regressão devido à sua interpretabilidade e capacidade de lidar com conjuntos de dados complexos.

~~~python
# Passo 1: Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = DecisionTreeClassifier() # Criar o modelo de Àrvore de Decisão
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("accuracy: ", accuracy)
~~~

![cla_3](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/7512e6a8-aac5-4ad5-b83a-279264afe2f1)

## Modelo de Regressão
As técnicas de regressão em Data Mining são métodos e algoritmos utilizados para modelar e prever relações entre variáveis, geralmente envolvendo a previsão de um valor numérico com base em outras variáveis conhecidas. Abaixo estão algumas das técnicas de regressão mais comuns:

* Regressão Linear Simples;
* Regressão Linear Múltipla;
* Regressão Logística;
* Regressão Polinomial;
* Métodos de Regressão Não Linear.

### Regressão Linear Simples

A regressão linear simples é um modelo estatístico que nos ajuda a entender a relação entre duas variáveis: uma variável dependente (geralmente representada como Y) e uma variável independente (geralmente representada como X).

Exemplo: Previsão de notas de exame com base no número de horas estudadas.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## Dados de horas de estudo e notas do exame
horas_estudo = np.array([2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
notas_exames = np.array([65,70,75,80,85,90,95,100,105,110])

# Criar um modelo de regressão linear 
modelo = LinearRegression()

# Treinar o modelo
coef_angular = modelo.coef_[0]
coef_linear = modelo.intercept_

# Plotar os dados e a reta de regressão
plt.scatter(horas_estudo, noptas_exames, color='blue')
plt.plot(horas_estudo, modelo.predict(horas_estudo), color='red')
plt.title('Regressão Linear Simples')
plt.xlabel('Horas de estudo')
plt.ylabel('Notas no Exame')
plt.show()

# Fazer previsões com o modelo
horas_estudo_novo = np.array([[8]]) # Horas do estudo do novo aluno
nota_prevista = modelo.predict(horas_estudo_novo)
print("Nota prevista para {} horas de estudo: {:.2f}.format(horas_estudo_novo[0][0], nota_prevista[0]))
~~~

![reg_1](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/f494ce80-4359-4727-9d53-4e57720981a0)

### Regressão Linear Múltipla

A regressão linear múltipla nos permite modelar como várias variáveis independentes afetam uma variável dependente.

Exemplo: Previsão de notas de exame com base no número de horas estudadas e tempo de sono.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Dados de horas de estudo, tempo de sono e notas do exame
horas_estudo = np.array([2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
tempo_sono = np.array([7,6,5,6,7,8,9,8,7,6]).reshape(-1,1)
notas_exames = np.array([65,70,75,80,85,90,95,100,105,110])

# Criar um modelo de regressão linear 
modelo = LinearRegression()

# Combinação de horas de estudos e tempo de sono com variáveis independentes
x = np.concatenate((horas_estudo,tempo_sono), axis=1)

# Treinar o modelo
modelo.fit(x, notas_exames)

# Coeficientes do modelo
coef_angular = modelo.coef_
coef_linear = modelo.intercept_

# Plotar os dados em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(horas_estudo, tempo_sono, notas_exames, color='blue')

# Prever notas para o intervalo de horas de estudos e tempo de sono
x_test = np.array([[x,y] for x in range(2, 12) for y in range(5,10)])
nota_previstas = modelo.predict(x_test)

# Plotar o plano de regressão
x_surf, y_surf = np.meshgrid(range(2,12), range(5,10))
exog = np.colum_stack((x_surf.flatten(), y_surf.flatten()))
nota_previstas = modelo.predict(exog)
ax.plot_surface(x_surf, y_surf, nota_previstas.reshape(x_surf.shape), color='red', alpha=0.5)

ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Tempo de Sono')
ax.set_zlabel('Notas do Exame')

plt.show()
~~~

![reg_2](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/d923ffee-3bbe-4230-bc84-51741bb7837e)

### Regressão Logística
A regressão logística é usada para modelar a probabilidade de uma variável dependente categórica com base em uma ou mais variáveis independentes. Utilizaremos a base de dados Iris, um conjunto de dados clássico frequentemente utilizado em aprendizado de máquina.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import standarScaler

# Passo 1: Carregar o conjunto de dados iris
iris = load_iris()
x = iris.data[:, :2] # Apenas as duas primeiras características para visualização
y = iris.target

# Passo 2: Dividir o conjunto de dados em conjunto de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size-0.3,random_state=42)

# Passo 3: Pré-processamento (padronização)
scaler = StandScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Passo 4: Criar e treinar modelo de regressãologística
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Passo 5: Fazer previsões no conjunto de teste
y_pred = model.predict(x_test_scaled)

# Passo 6: Avaliar o desempenho do modelo
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n, classification_report(y_test,y_pred))

# Passo 7: Visiuazlização dos resultados
plt.figure(figsize=(10,6))

# Plotar os pontos de dados de treinamento
plt.scatter(x_train_scaled[:,0], x_test_scaled[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Treinamento')

# Plotar os pontos de dados de teste
plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_train, cmap='viridis', marker='x', s='100', label='Teste')

# Plotar as regiões de decisão
x_min, x_max = x_train_scaled[:, 0].min() -1, x_train_scaled[:,0].max() + 1
y_min, y_max = x_train_scaled[:, 0].min() -1, x_train_scaled[:,0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, alpha=0.3, cmap='viridis')

plt.xlabel('Comprimento da Sépala Padronizado')
plt.ylabel('Largura da Sépala Padronizado')
plt.title('Regressão Logística para Classificaçãode Espécies Iris')
plt.legend()
plt.show()
~~~

![reg_3](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/5ef97532-452d-435a-a4c6-a80998d12f50)
![reg_4](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/9c9a8daf-acc0-415d-b8f9-daed961c6421)

### Regressão Polinomial
A regressão polinomial é uma extensão da regressão linear que permite modelar relacionamentos não lineares entre as variáveis independentes e dependentes.

Exemplo: Ajustar uma regressão polinomial aos dados.

~~~python
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerar dados sintéticos
np.random.seed(0)
x = 2 * np.random.rand(100, 1) -1 # Variáveis independentes entre -1 e 1
y = 3 * x**2 + 0.5 * x + 2 + np.random.randn(100, 1) # Relação quadrática com ruído

# Plotar os dados
plt.scatter(x,y, color='blue', label='Dados')

# Ajustar uma regressão polimonial de grau 2 aos dados
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# Plotar a curva ajustada
x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = poly_features.transform(x_plot)
y_plot = lin_reg.predict(x_plot_poly)
plt.plot(x_plot, y_plot, color='red', label='Regressão Polimonial (grau 2) ')

# Avaliar Modelo
y_pred = lin_reg.predict(x_poly)
mse = mean_squared_error(y,y_pred)
print("Erro médio quadrático:", mse)

plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Polimonial de Grau 2')
plt.legend()
plt.show()
~~~

![reg_5](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/e3827567-92ee-414a-b919-73442f46d7d8)

### Métodos de Regressão Não Linear
A regressão não linear é uma técnica usada para modelar relacionamentos complexos entre variáveis independentes e dependentes, onde a relação não pode ser representada por uma função linear simples. Diferentemente da regressão linear, que assume uma relação linear entre as variáveis, a regressão não linear permite que o modelo se ajuste a padrões mais complexos nos dados.

Exemplo: Ajustar um modelo exponencial aos dados.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Função exponencial para ajustar aos dados
def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)
    
# Gerar dados sintéticos
no.random.seed(0)
x = np.linspace(0, 5, 100) # Variável independente
y = 2.5 * np.exp(0.5 * x) + np.random.normal(0, 0.5,100) # Relação exponencial com ruído

# Ajustar o modelo aos dados usando o curve_fit
params, _ = curve_fit(modelo_exponencial,x,y)

# Plotar os dados
plt.scatter(x,y, color='blue', label='Dados')

# Plotar a curva ajustada
plt.plot(x, modelo_exponencial(x, *params), color='red', label='Regressão Exponencial')

plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Não Linear Exponencial')
plt.legend()
plt.show()
~~~

![reg_6](https://github.com/Jidsx/Exerc_Classificacao_Regressao/assets/113401757/e8af0068-9bb9-4adf-b1ab-ac6bd405159e)
