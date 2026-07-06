# Rede Neural de Classificação com TensorFlow.js

Um exemplo didático de rede neural construída com [TensorFlow.js para Node](https://www.tensorflow.org/js) que classifica pessoas em três categorias (**premium**, **medium**, **basic**) a partir de três características: idade, cor favorita e localização.

O objetivo do projeto é **aprender os conceitos fundamentais** de uma rede neural supervisionada: preparação de dados, normalização, one-hot encoding, treinamento e predição.

---

## Sumário

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como executar](#como-executar)
- [Como funciona](#como-funciona)
  - [1. Preparação dos dados](#1-preparação-dos-dados)
  - [2. Arquitetura da rede](#2-arquitetura-da-rede)
  - [3. Treinamento](#3-treinamento)
  - [4. Predição](#4-predição)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Conceitos-chave](#conceitos-chave)
- [Próximos passos](#próximos-passos)

---

## Pré-requisitos

- **Node.js** 18 ou superior
- **npm** (já vem com o Node)

## Instalação

```bash
npm install
```

Isso vai instalar o `@tensorflow/tfjs-node` (versão 4.22), a única dependência do projeto.

## Como executar

Para rodar uma única vez:

```bash
npm run run
```

Para rodar em modo de desenvolvimento (re-executa a cada alteração no arquivo):

```bash
npm start
```

Saída esperada (aproximada — varia a cada execução por causa da inicialização aleatória dos pesos):

```
basic (87.34%)
medium (10.21%)
premium (2.45%)
```

---

## Como funciona

### 1. Preparação dos dados

A rede neural só entende **números**, então os dados originais precisam ser convertidos.

**Dados de treino (conceituais):**

| Nome   | Idade | Cor      | Localização |
|--------|-------|----------|-------------|
| Erick  | 30    | azul     | São Paulo   |
| Ana    | 25    | vermelho | Rio         |
| Carlos | 40    | verde    | Curitiba    |

Essas pessoas são convertidas em vetores de **7 posições** cada:

| Posição | 0                 | 1    | 2        | 3     | 4         | 5   | 6         |
|---------|-------------------|------|----------|-------|-----------|-----|-----------|
| Campo   | idade normalizada | azul | vermelho | verde | São Paulo | Rio | Curitiba  |

#### Normalização (posição 0)

A idade é normalizada para um valor entre 0 e 1 usando **min-max**:

```
idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
```

Com `idade_min = 25` e `idade_max = 40`:

- Erick (30): `(30 - 25) / (40 - 25) = 0.33`
- Ana (25): `0`
- Carlos (40): `1`

#### One-hot encoding (posições 1-6)

Categorias viram colunas binárias — `1` na categoria correta, `0` nas outras. Isso evita que a rede infira uma ordem inexistente (por ex. `azul < vermelho < verde`).

**Tensor de entrada final:**

```js
const normalizedPeopleTensor = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick   → azul, São Paulo
    [0,    0, 1, 0, 0, 1, 0], // Ana     → vermelho, Rio
    [1,    0, 0, 1, 0, 0, 1]  // Carlos  → verde, Curitiba
]
```

**Tensor de labels** (também em one-hot, na ordem `[premium, medium, basic]`):

```js
const labelsTensor = [
    [1, 0, 0], // Erick  → premium
    [0, 1, 0], // Ana    → medium
    [0, 0, 1]  // Carlos → basic
];
```

### 2. Arquitetura da rede

```
Entrada (7)  →  Hidden Dense (80, ReLU)  →  Saída Dense (3, Softmax)
```

- **Camada oculta** — 80 neurônios com ativação **ReLU**. ReLU "filtra" valores: deixa passar números positivos e zera o resto, permitindo que a rede aprenda padrões não-lineares.
- **Camada de saída** — 3 neurônios com ativação **Softmax**, que transforma os valores em **probabilidades que somam 1** (uma probabilidade por categoria).

### 3. Treinamento

A rede é compilada com:

- **Optimizer:** `adam` — ajusta os pesos de forma adaptativa, aprendendo com o histórico de erros.
- **Loss:** `categoricalCrossentropy` — adequada para classificação multi-classe com saída em one-hot. Penaliza a rede proporcionalmente à distância entre a previsão e a resposta correta.
- **Métrica:** `accuracy` — porcentagem de acertos.

Configurações de treino:

- **`epochs: 100`** — passa pelo dataset 100 vezes
- **`shuffle: true`** — embaralha os dados a cada época para evitar viés
- **`verbose: 0`** — sem log interno

### 4. Predição

Uma nova pessoa (ex: `{ nome: 'zé', idade: 28, cor: 'verde', localizacao: 'Curitiba' }`) é convertida no mesmo formato:

```js
const normalizedPersonTensor = [
    [0.2, 1, 0, 0, 0, 1, 0]
    //  ↑   ↑─cores─↑  ↑─localizações─↑
    //  idade normalizada
]
```

A rede produz um vetor de probabilidades, ex: `[0.05, 0.10, 0.85]`. O código ordena pela maior probabilidade e usa `labelNames` para traduzir os índices de volta para nomes legíveis.

---

## Estrutura do projeto

```
exemplo-00/
├── index.js          # Todo o código: dados, treino e predição
├── package.json      # Dependências e scripts
├── package-lock.json
└── README.md
```

---

## Conceitos-chave

| Conceito                    | O que é                                                                |
|-----------------------------|------------------------------------------------------------------------|
| **Normalização**            | Trazer valores numéricos para uma escala comum (0 a 1)                 |
| **One-hot encoding**        | Converter categorias em vetores binários sem criar ordem implícita     |
| **Camada densa (Dense)**    | Cada neurônio se conecta a todos da camada anterior                    |
| **ReLU**                    | Ativação que zera valores negativos; introduz não-linearidade          |
| **Softmax**                 | Converte logits em probabilidades que somam 1                          |
| **Optimizer Adam**          | Algoritmo que ajusta os pesos da rede de forma adaptativa              |
| **Categorical Crossentropy**| Função de perda para classificação multi-classe com labels one-hot     |
| **Epoch**                   | Uma passagem completa pelo dataset durante o treinamento               |

---

## Próximos passos

Algumas ideias para evoluir o exemplo:

- Aumentar o dataset (com 3 pessoas a rede não generaliza bem — `quanto mais dado melhor!`)
- Separar dados de **treino** e **teste** para medir a acurácia real
- Adicionar uma **camada de dropout** para reduzir overfitting
- Externalizar a normalização e o one-hot encoding em funções reutilizáveis
- Salvar o modelo treinado com `model.save()` e carregá-lo depois
- Experimentar diferentes números de neurônios, épocas e funções de ativação
