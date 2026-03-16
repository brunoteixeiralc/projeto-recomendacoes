import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
import { normalize } from '../utils/math.js';
import { makeContext, oneHotWeighted } from '../utils/dataProcessor.js';
// Importação do Supabase via CDN (para ambiente de Worker/Browser)
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.42.0/+esm';

// Log para confirmar que o arquivo do Worker foi carregado com sucesso
console.log('Model training worker initialized');

// ============================================================================
// CLASSE PRINCIPAL (Isola todo o Escopo e a Inteligência da Aplicação)
// ============================================================================

class RecommendationEngine {
    constructor() {
        // Inicializa o contexto global vazio (onde guardaremos metadados e vetores)
        this.globalCtx = {};
        // O modelo da rede neural começa como nulo até o treinamento
        this.model = null;

        // Pesos de importância para cada característica na hora de calcular recomendações
        this.WEIGHT = {
            category: 0.4, // Categoria é muito importante (40% de peso)
            color: 0.3,    // Cor tem 30% de peso
            price: 0.2,    // Preço tem 20% de peso
            age: 0.1,      // Idade tem 10% de peso
            gender: 0.5,   // Gênero é o fator mais forte (50% de peso extra)
        };

        // Cliente do Supabase começa nulo até receber as chaves via mensagem
        this.supabase = null;
    }

    /**
     * Inicializa a conexão com o Supabase usando URL e Key seguras
     */
    initSupabase({ url, key }) {
        // Valida se as credenciais foram enviadas corretamente
        if (!url || !key) {
            console.error('Supabase URL and Key are required for initialization');
            return;
        }
        // Armazena as credenciais na instância
        this.supabaseUrl = url;
        this.supabaseKey = key;
        // Cria o cliente oficial do Supabase
        this.supabase = createClient(this.supabaseUrl, this.supabaseKey);
        console.log('Supabase client initialized securely');
    }

    /**
     * Função principal de treinamento: baixa dados, prepara contexto e treina a IA
     */
    async trainModel({ users }) {
        console.log('Training model with users:', users)

        // Envia uma mensagem para a UI indicando que o processo começou (50% de carga inicial)
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

        // Busca o catálogo de produtos no servidor e converte para JSON
        const products = await fetch('/data/products.json').then(res => res.json());

        // Processa os dados brutos de usuários e produtos para criar metadados matemáticos
        const context = makeContext(users, products);

        // Gera e armazena os vetores (embeddings) matemáticos de cada produto do catálogo
        context.productsVector = products.map(product => {
            return {
                name: product.name,
                meta: { ...product },
                // Converte o produto em números e extrai do tensor para um formato de dados
                vector: this.encodeProduct(product, context).dataSync()
            }
        })

        // Guarda o contexto completo (metadados + vetores) para consultas futuras
        this.globalCtx = context

        // Se o Supabase estiver configurado, envia os vetores gerados para o banco de dados
        if (this.supabase) {
            await this.syncProductsToSupabase(context.productsVector);
        } else {
            console.warn('Supabase not initialized. Skipping vector sync.');
        }

        // Prepara o conjunto de dados para treinamento (Xs e Ys)
        const trainData = this.createTrainingData(context)
        // Configura as camadas da Rede Neural e inicia o processo de aprendizado
        this.model = await this.configureNeuralNetAndTrain(trainData)

        // Informa a thread principal que o treinamento terminou com sucesso
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }

    /**
     * Gera recomendações comparando o "DNA" (vetor) do usuário com o DNA dos produtos.
     */
    recommend(user) {
        // Impede a recomendação se a IA ainda não tiver aprendido nada (modelo nulo)
        if (!this.model) {
            console.error('Model not trained yet')
            return
        }

        // Pega o contexto global calculado no treinamento
        const context = this.globalCtx

        // Transforma o perfil do usuário atual em um vetor numérico (DNA do usuário)
        const userVector = this.encodeUser(user, context).dataSync()

        // Para cada produto, cria uma dupla [DNA_Usuario, DNA_Produto] para a rede neural analisar
        const inputs = context.productsVector.map(vector => {
            return [...userVector, ...vector.vector]
        })

        // Converte o array de duplas em uma matriz 2D (Tensor) para processamento rápido
        const inputTensor = tf.tensor2d(inputs)

        // Pede para o modelo "prever" o interesse do usuário em cada um dos produtos
        const predictions = this.model.predict(inputTensor)

        // Extrai as predições (números entre 0 e 1) do formato Tensor
        const score = predictions.dataSync()

        // Associa cada pontuação da IA ao respectivo nome e detalhes do produto
        const recommendations = context.productsVector.map((product, index) => {
            return {
                ...product.meta,
                name: product.name,
                score: score[index] // Pontuação de afinidade dada pelo modelo
            }
        })

        // Coloca os produtos mais interessantes (maior score) no topo da lista
        const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score)

        // Devolve a lista final de recomendações para a interface aparecer para o usuário
        postMessage({
            type: workerEvents.recommend,
            user,
            recommendations: sortedRecommendations
        });
    }

    /**
     * Envia os vetores calculados para o Supabase (Vector Database).
     */
    async syncProductsToSupabase(productsVector) {
        console.log('Syncing products to Supabase...');
        
        // Mapeia os dados para o formato que a tabela 'products' espera
        const productsToUpsert = productsVector.map(pv => ({
            name: pv.name,
            category: pv.meta.category,
            color: pv.meta.color,
            price: pv.meta.price,
            embedding: Array.from(pv.vector) // Transforma o Tensor num array JS normal (necessário para o banco)
        }));

        // Faz o "Upsert": Se o produto existe (pelo nome), atualiza. Se não, insere um novo.
        const { error } = await this.supabase
            .from('products')
            .upsert(productsToUpsert, { onConflict: 'name' });

        // Loga erro ou sucesso na console para depuração
        if (error) {
            console.error('Error syncing to Supabase:', error.message);
        } else {
            console.log('Products synced successfully!');
        }
    }

    // ============================================================================
    // MODELO / INTELIGÊNCIA ARTIFICIAL
    // ============================================================================

    /**
     * Configura o "Cérebro" (Rede Neural) e faz ele estudar os dados de compra.
     */
    async configureNeuralNetAndTrain(trainData) {
        // Inicia uma pilha de camadas ("Sequential")
        const model = tf.sequential()

        // Camada de entrada com 128 neurônios artificiais
        model.add(tf.layers.dense({
            inputShape: [trainData.inputDimensions], // Tamanho do vetor combinado
            units: 128,                              
            activation: 'relu'                       // Ativação ReLU (ignora valores negativos)
        }))

        // Camada oculta intermediária com 64 neurônios
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }))

        // Outra camada oculta com 32 neurônios (vai afunilando o conhecimento)
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }))

        // Camada final: gera apenas 1 saída entre 0 e 1 (Sigmoid)
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }))

        // Define as regras de aprendizado: Otimizador Adam e cálculo de erro Cross-Entropy
        model.compile({
            optimizer: tf.train.adam(0.01),      // Taxa de aprendizado de 0.01
            loss: 'binaryCrossentropy',          // Ideal para decisões Sim/Não
            metrics: ['accuracy']                // Queremos ver a precisão em %
        })

        // Inicia o treinamento real (estudo dos dados)
        await model.fit(trainData.xs, trainData.ys, {
            epochs: 100,                         // O modelo lê os dados 100 vezes
            batchSize: 32,                       // Estudando grupos de 32 exemplos
            shuffle: true,                       // Embaralha para não viciar na ordem
            callbacks: {
                // A cada "rodada" (época), avisa a UI sobre como está a precisão e o erro
                onEpochEnd: (epoch, logs) => {
                    postMessage({
                        type: workerEvents.trainingLog,
                        epoch: epoch,
                        loss: logs.loss,
                        accuracy: logs.acc
                    });
                }
            }
        })

        return model
    }

    // ============================================================================
    // FUNÇÕES AUXILIARES / PREPARAÇÃO DE DADOS MATEMÁTICOS
    // ============================================================================

    /**
     * Transforma o histórico de compras e catálogo em dados que a IA entende (0s e 1s).
     */
    createTrainingData(context) {
        const inputs = [] // Exemplos de (Usuario + Produto)
        const labels = [] // Resultado real (Comprou = 1, Não Comprou = 0)

        // Filtra para usar apenas usuários que já fizeram alguma compra no passado
        context.users
            .filter(u => u.purchases.length)
            .forEach(user => {
                // Vetoriza o perfil do usuário
                const userVector = this.encodeUser(user, context).dataSync()

                // Compara cada usuário com todos os produtos possíveis
                context.products.forEach(product => {
                    // Vetoriza as características do produto
                    const productsVector = this.encodeProduct(product, context).dataSync()

                    // Verifica se no histórico real o usuário comprou esse item
                    const label = user.purchases.some(
                        purchase => purchase.name == product.name ? 1 : 0
                    )

                    // Junta o vetor do usuário com o do produto e guarda o resultado
                    inputs.push([...userVector, ...productsVector])
                    labels.push(label)
                })
            });

        // Converte as listas gigantes em estruturas otimizadas para IA (Tensores)
        return {
            xs: tf.tensor2d(inputs),                              // Matriz de características
            ys: tf.tensor2d(labels, [labels.length, 1]),         // Matriz de resultados
            inputDimensions: context.dimensions * 2               // Total de neurônios de entrada
        }
    }

    /**
     * Transforma um objeto Usuário em um código matemático (Embedding).
     */
    encodeUser(user, context) {
        // Transforma o gênero (texto) em um vetor numérico ponderado pelo peso do gênero
        const genderVector = oneHotWeighted(
            tf,
            context.genderIndex[user.gender] ?? 0,
            context.numGenders,
            this.WEIGHT.gender
        )

        // Caso o usuário já tenha compras: baseia o perfil na "média" do que ele gosta
        if (user.purchases.length) {
            // Cria uma média dos vetores dos produtos que ele já comprou
            const productMeans = tf.stack(
                user.purchases.map(
                    product => this.encodeProduct(product, context)
                )
            )
            .mean(0) // Calcula a média rastro por rastro

            // Remove a parte neutra de gênero que vem dos produtos
            const vectorWithoutGender = productMeans.slice([0], [context.dimensions - context.numGenders])

            // Coloca o gênero real do usuário no lugar para formar o vetor completo
            return tf.concat1d([vectorWithoutGender, genderVector])
                .reshape([1, context.dimensions])
        }

        // Caso usuário novo (Cold Start): Baseado apenas em Idade e Gênero
        return tf.concat1d([
            tf.zeros([1]), // Sem histórico de preço
            tf.tensor1d([
                normalize(
                    user.age,
                    context.minAge,
                    context.maxAge
                ) * this.WEIGHT.age
            ]), // Idade normalizada por peso
            tf.zeros([context.numCategories]), // Sem preferências de categoria ainda
            tf.zeros([context.numColors]),     // Sem preferências de cor ainda
            genderVector                       // Gênero real do usuário
        ]).reshape([
            1,
            context.dimensions
        ])
    }

    /**
     * Converte as características de um Produto (cor, preço, categoria) em números.
     */
    encodeProduct(product, context) {
        // Normaliza o preço para a escala 0 a 1 e aplica o peso definido
        const price = tf.tensor1d([
            normalize(product.price, context.minPrice, context.maxPrice) * this.WEIGHT.price
        ])
        
        // Pega a idade média do público desse produto e normaliza
        const age = tf.tensor1d([
            (context.productAvgAgeNorm[product.name] ?? 0.5) * this.WEIGHT.age
        ])

        // Transforma Categoria e Cor (textos) em posições em um vetor (One-Hot) com seus respectivos pesos
        const category = oneHotWeighted(
            tf, context.categoryIndex[product.category], context.numCategories, this.WEIGHT.category
        )
        const color = oneHotWeighted(
            tf, context.colorIndex[product.color], context.numColors, this.WEIGHT.color
        )

        // Produtos são "neutros" em gênero, então preenchemos com zeros no final
        const genderPlaceholder = tf.zeros([context.numGenders])

        // Junta todas as partes num único vetor que define a identidade do produto para a IA
        return tf.concat1d([price, age, category, color, genderPlaceholder])
    }
}

// ============================================================================
// INSTANCIAÇÃO E ENTRY-POINT (Porta de Entrada do Worker)
// ============================================================================

// Cria a instância do "motor" que acabamos de definir acima
const engine = new RecommendationEngine();

// Lista de ações que a thread principal pode pedir para o Worker fazer
const handlers = {
    // Inicialização do banco de dados (Supabase)
    'init': d => engine.initSupabase(d.config),
    // Pedido para iniciar o treinamento
    [workerEvents.trainModel]: engine.trainModel.bind(engine),
    // Pedido para recomendar algo para alguém
    [workerEvents.recommend]: d => engine.recommend(d.user),
};

// Escutador oficial de mensagens: quando chega uma mensagem, decide qual ação executar
self.onmessage = e => {
    const { action, ...data } = e.data;
    // Se comando for reconhecido, executa passando os dados recebidos
    if (handlers[action]) handlers[action](data);
};

