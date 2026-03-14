<p align="center">
  <h1 align="center">🛍️ E-commerce Recommendation System with ML</h1>
</p>

<p align="center">
  <strong>Uma simulação de E-commerce com Sistema de Recomendação embutido no lado do cliente usando TensorFlow.js.</strong>
</p>

<p align="center">
  <img alt="JavaScript" src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E">
  <img alt="TensorFlow.js" src="https://img.shields.io/badge/TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img alt="HTML5" src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white">
  <img alt="CSS3" src="https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white">
  <img alt="Jest" src="https://img.shields.io/badge/-jest-%23C21325?style=for-the-badge&logo=jest&logoColor=white">
</p>

<hr>

## 📖 Sobre o Projeto
Este projeto experimental é um e-commerce em Vanilla JS com uma arquitetura modularizada (quase-MVC: Models/Data, Views, Controllers, Services e Workers). O seu grande diferencial é a **construção e treinamento on-the-fly de um modelo de Machine Learning** usando o `TensorFlow.js`. 

O objetivo final do modelo é rodar em um Web Worker para fazer o *One-Hot Encoding* de variáveis como preços, categorias, cores e idades visando "aprender" o perfil do público de cada produto para **recomendar de forma personalizada novos produtos baseados na intersecção do que o usuário já comprou**.

## ✨ Funcionalidades
- **Seleção de Perfil**: O usuário simula o acesso com um perfil (extraído de um `.json` de usuários) para acessar a plataforma.
- **Histórico de Compras Tracking**: Rastreio do que o usuário clica e tenta "comprar", persistido em memória/session.
- **Arquitetura Modular**: Implementação robusta de Controllers, Views independentes e injeção/comunicação guiada por Eventos.
- **Isolamento de Estado (OOP)**: O Worker que treina a IA utiliza o paradigma Orientado a Objetos (`RecommendationEngine`) para evitar vazamento do escopo global.
- **Inteligência Artificial**: Treinamento de Redes Neurais acontecendo silenciosamente em Background usando Service Workers sem travar a interface da aplicação (`src/workers/modelTrainingWorker.js`).

## 🚀 Como Rodar Localmente

Certifique-se de ter o [Node.js](https://nodejs.org/) instalado.

1. Clone o repositório
   ```bash
   git clone https://github.com/SEU_USUARIO/projeto-recomendacoes.git
   ```

2. Entre no diretório
   ```bash
   cd ecommerce-recomendations-with-ml
   ```

3. Instale as dependências (Browser-sync)
   ```bash
   npm install
   ```

4. Inicie o servidor local
   ```bash
   npm start
   ```

A página será aberta automaticamente via Browser-Sync (geralmente na porta `3000`), fazendo _live-reloading_ caso algum arquivo seja alterado.

## 🧪 Testes Automatizados

O projeto conta com testes unitários focados na confiabilidade de funções matemáticas essenciais para a Engenharia de Features do TensorFlow, como normalização (`math.js`) e mapeamento do contexto de produtos baseados no histórico mockado (`dataProcessor.js`).

Para rodar a suíte de testes com **Jest**:

```bash
npm run test
```

## 🗂 Estrutura de Diretórios
```bash
.
├── data/                    # JSONs estruturados servindo como "Banco de Dados" Local
│   ├── products.json        # Catálogo do Ecommerce
│   └── users.json           # Usuários mockados c/ histórico de idades e compras
├── src/
│   ├── controller/          # Regras de orquestração de Views e Services
│   ├── events/              # Lógica de Pub/Sub e eventos internos  
│   ├── service/             # Requisições Fetch assíncronas aos JSONs
│   ├── utils/               # Utilitários (como dataProcessor.js e math.js)
│   ├── view/                # Manipulação de DOM / HTML 
│   ├── workers/             # Web Workers (Onde a classe RecommendationEngine mora)
│   └── index.js             # Entrypoint da Aplicação 
├── index.html               # Estrutura Main HTML
├── style.css                # Estilizacão Global
└── package.json            
```

## 🧠 Lógica do TensorFlow.js (Passo a Passo)

A inteligência do sistema reside no `modelTrainingWorker.js`, funcionando como um "cérebro isolado" em Background. Abaixo está o fluxo detalhado:

### 1. Inicialização e Estrutura
O processamento ocorre em um **Web Worker**, garantindo que a thread principal (UI) nunca trave durante cálculos pesados. A lógica é encapsulada na classe `RecommendationEngine`, que gerencia o estado do modelo e os pesos das características (Preço, Categoria, Cor e Idade).

Antes do treino, o sistema realiza o pré-processamento:
- **Carga:** Consome o catálogo de produtos e usuários.
- **Vetorização (Embeddings):** Transforma produtos em vetores numéricos. Variáveis categóricas sofrem *One-Hot Encoding* e valores contínuos (preço/idade) são normalizados entre 0 e 1.
- **Otimização:** Os vetores dos produtos são pré-calculados e armazenados em memória (`productsVector`) para acelerar as recomendações posteriores.

### Passo a Passo da Lógica (TensorFlow.js)

1.  **Inicialização do Motor:** O sistema carrega o catálogo de produtos e os perfis de usuários (com idade e sexo).
2.  **Engenharia de Features (Vetorização):**
    *   **Produtos:** São convertidos em vetores baseados em preço, cor, categoria e idade média dos compradores.
    *   **Usuários:** O perfil é criado pela média dos produtos comprados **somada ao seu gênero (sexo)**.
3.  **Rede Neural (Treinamento):** O modelo aprende a relação entre o vetor do usuário e a probabilidade de compra de cada produto.
4.  **Ciclo de Recomendação:**
    *   **Usuários Antigos:** A rede usa o histórico e o gênero para sugerir produtos parecidos.
    *   **Novos Usuários (Cold Start):** Em vez de recomendações genéricas, o sistema usa a **idade e o sexo** informados para gerar uma predição personalizada imediata.
5.  **Comunicação:** Tudo roda em um Web Worker, garantindo que o site não trave durante o processamento da IA.
atálogo simultaneamente.
- **Ranking:** Os produtos são ordenados pelo *score* de afinidade devolvido pela IA e enviados de volta para a interface.

### 5. Comunicação Assíncrona
O Worker utiliza um sistema de `handlers` para responder a eventos (`trainModel`, `recommend`) e reporta logs de progresso e acurácia em tempo real para o console e interface.
