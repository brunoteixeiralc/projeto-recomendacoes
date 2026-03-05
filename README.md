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

## 🧠 Lógica do TensorFlow.js
A magia das recomendações acontecerá no `modelTrainingWorker.js`, o qual:
1. Normaliza as idades e preços para Valores entre 0 e 1 calculando a "Média Ponderada" do público de um produto (`(Max - Min) / 2`).
2. Faz transformações de *One-Hot Encoding* convertendo variáveis textuais categóricas (como *Azul*, *Verde*, *Eletrônicos*, *Vestuário*) em Múltiplas Dimensões/Arrays Binários, calculando também automaticamente o número de Dimensões a injetar na Rede Neural.
3. Roda em uma Thread fora da janela principal (Web Worker) para que os eventuais lags de processamento/treino do Tensor não "engasquem" a interface de CSS e navegação do cliente.
