<p align="center">
  <h1 align="center">🛍️ Projeto de recomendações</h1>
</p>

<p align="center">
  <strong>Sistema de recomendação inteligente que personaliza a vitrine do e-commerce em tempo real, sem impactar a performance da interface.</strong>
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
- **Isolamento de Estado (OOP):** O Worker de treinamento da IA utiliza Programação Orientada a Objetos (`RecommendationEngine`) para evitar vazamento de escopo global.
- **Inteligência Artificial:** O treinamento da rede neural ocorre silenciosamente em background usando Web Workers sem travar a interface (`src/workers/modelTrainingWorker.js`).
- **Banco de Dados Vetorial (Supabase):** Integração com o Supabase usando `pgvector` para persistência e busca semântica de embeddings dos produtos.
- **Internacionalização (i18n):** Suporte nativo a múltiplos idiomas (PT-BR e EN) com troca dinâmica na UI e persistência de preferência.
- **Conversão de Moeda Dinâmica:** Conversão de preços em tempo real (USD para BRL) utilizando a **AwesomeAPI**, sincronizada com o idioma selecionado.
- **Segurança Proativa:** Gerenciamento de credenciais sensíveis via `src/config.js` (ignorado pelo Git) e comunicação segura com o Worker.

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

3. Instale as dependências (Browser-sync, Jest, Cross-env)
   ```bash
   npm install
   ```

4. Configuração do Supabase
   - Crie o arquivo `src/config.js` baseado no modelo de código para incluir sua `supabaseUrl` e `supabaseKey`.
   - Rode o script SQL de configuração disponível no banco.

5. Inicie o servidor local
   ```bash
   npm start
   ```

A página será aberta automaticamente via Browser-Sync (geralmente na porta `3000`), fazendo _live-reloading_ caso algum arquivo seja alterado.

## 🧪 Testes Automatizados

O projeto conta com testes unitários focados na confiabilidade de funções matemáticas essenciais para a Engenharia de Features do TensorFlow, como normalização (`math.js`) e mapeamento do contexto de produtos baseados no histórico mockado (`dataProcessor.js`). Também testamos a comunicação do `WorkerController`.

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
│   ├── config.js            # Configurações sensíveis (Ignorado pelo Git)
│   └── index.js             # Entrypoint da Aplicação 
├── index.html               # Estrutura Main HTML
├── style.css                # Estilizacão Global
└── package.json            
```

## 🧠 Lógica do TensorFlow.js (Passo a Passo)

A inteligência do sistema reside no `modelTrainingWorker.js`, funcionando como um "cérebro isolado" em Background. Abaixo está o fluxo detalhado:

### 1. Inicialização e Estrutura
O processamento ocorre em um **Web Worker**, garantindo que a thread principal (UI) nunca trave durante cálculos pesados. A lógica é encapsulada na classe `RecommendationEngine`, que gerencia o estado do modelo e os pesos das características (Preço, Categoria, Cor e Idade).

### 2. Vetorização e Sincronização
- **Embeddings:** Transforma produtos em vetores numéricos de 16 dimensões.
- **Sincronização:** Após o cálculo, os vetores são enviados para o **Supabase** via `upsert`. Isso permite que o conhecimento da IA seja persistente e possa ser consultado via SQL usando busca por similaridade de cosseno (`pgvector`).

### 3. Passo a Passo da Lógica
1.  **Inicialização do Motor:** O sistema carrega o catálogo de produtos e os perfis de usuários.
2.  **Engenharia de Features:**
    *   **Produtos:** Convertidos em vetores baseados em preço, cor, categoria e idade média.
    *   **Usuários:** O perfil é criado pela média dos produtos comprados somada ao seu gênero.
3.  **Rede Neural (Treinamento):** O modelo aprende a relação entre o vetor do usuário e a probabilidade de compra.
4.  **Ciclo de Recomendação:**
    *   **Usuários Antigos:** A rede usa histórico e gênero para predição.
    *   **Novos Usuários (Cold Start):** O sistema usa a idade e o sexo informados para gerar uma predição imediata.

### 4. Internacionalização e Moeda
- **Tradução:** O sistema utiliza o `TranslationService` para traduzir não apenas os selos da interface, mas também os dados dinâmicos (nomes de produtos, categorias e cores).
- **Câmbio:** O `CurrencyService` consome a AwesomeAPI para obter a cotação do dólar em tempo real, formatando os preços conforme o local (`en-US` ou `pt-BR`).

### 5. Comunicação
O Worker reporta logs de progresso e acurácia em tempo real para o console e interface através de uma ponte segura controlada pelo `WorkerController`.
