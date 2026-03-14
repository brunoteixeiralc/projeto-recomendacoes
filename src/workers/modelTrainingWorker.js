import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
import { normalize } from '../utils/math.js';
import { makeContext, oneHotWeighted } from '../utils/dataProcessor.js';

console.log('Model training worker initialized');

// ============================================================================
// CLASSE PRINCIPAL (Isola todo o Escopo e a Inteligência da Aplicação)
// ============================================================================

class RecommendationEngine {
    constructor() {
        this.globalCtx = {};
        this.model = null;

        this.WEIGHT = {
            category: 0.4,
            color: 0.3,
            price: 0.2,
            age: 0.1,
        };
    }

    async trainModel({ users }) {
        console.log('Training model with users:', users)

        // Envia uma mensagem de volta para o resto da aplicação indicando que o processo está na metade (50%)
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

        // Baixa o arquivo 'products.json' e converte a resposta de JSON para um array Javascript
        const products = await fetch('/data/products.json').then(res => res.json());

        // Cria o "contexto" usando a nossa função makeContext() que extrai cálculos matemáticos da base
        const context = makeContext(users, products);

        // Pré-calcula e armazena os vetores (embeddings) de todos os produtos do catálogo.
        // Isso otimiza a performance, pois evitaremos calcular os mesmos vetores repetidamente durante a recomendação.
        context.productsVector = products.map(product => {
            return {
                name: product.name,
                meta: { ...product },
                vector: this.encodeProduct(product, context).dataSync()
            }
        })

        // Salva todo esse contexto calculado para uso isolado do Worker
        this.globalCtx = context

        const trainData = this.createTrainingData(context)
        this.model = await this.configureNeuralNetAndTrain(trainData)

        // Envia mensagem indicando progresso e finalização
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }

    /**
     * Gera recomendações para um usuário específico comparando o seu perfil com todos os produtos.
     */
    recommend(user) {
        // Verifica se o modelo já foi treinado antes de tentar recomendar
        if (!this.model) {
            console.error('Model not trained yet')
            return
        }

        // Recupera o contexto global que contém os produtos pré-vetorizados
        const context = this.globalCtx

        // Converte o usuário atual em um vetor representativo (embedding)
        const userVector = this.encodeUser(user, context).dataSync()

        // Para cada produto disponível, cria uma entrada combinada: [vetor_do_usuario, vetor_do_produto]
        const inputs = context.productsVector.map(vector => {
            return [...userVector, ...vector.vector]
        })

        // Converte a lista de entradas em um Tensor 2D para processamento na rede neural
        const inputTensor = tf.tensor2d(inputs)

        // Realiza a predição: a rede retorna a probabilidade (score) de o usuário gostar de cada produto
        const predictions = this.model.predict(inputTensor)

        // Extrai os valores numéricos das predições
        const score = predictions.dataSync()

        // Mapeia os scores de volta para os metadados dos produtos originais
        const recommendations = context.productsVector.map((product, index) => {
            return {
                ...product.meta,
                name: product.name,
                score: score[index] // Atribui o score de afinidade calculado
            }
        })

        // Ordena as recomendações da maior afinidade para a menor
        const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score)

        // Envia os resultados ordenados de volta para a thread principal da aplicação
        postMessage({
            type: workerEvents.recommend,
            user,
            recommendations: sortedRecommendations
        });
    }

    // ============================================================================
    // MODELO / INTELIGÊNCIA ARTIFICIAL
    // ============================================================================

    /**
     * Configura a arquitetura da rede neural e realiza o treinamento.
     */
    async configureNeuralNetAndTrain(trainData) {
        // Cria um modelo sequencial, onde as camadas são empilhadas uma após a outra
        const model = tf.sequential()

        // Primeira camada: Densa (totalmente conectada)
        model.add(tf.layers.dense({
            inputShape: [trainData.inputDimensions], // Formato da entrada (vetor de usuário + vetor de produto)
            units: 128,                              // 128 neurônios nesta camada
            activation: 'relu'                       // Função de ativação ReLU para introduzir não-linearidade
        }))

        // Segunda camada oculta: 64 neurônios
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }))

        // Terceira camada oculta: 32 neurônios
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }))

        // Camada de saída: 1 neurônio com ativação Sigmoid (retorna um valor entre 0 e 1, ideal para classificação binária)
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }))

        // Compila o modelo definindo o otimizador (Adam), a função de perda e a métrica de avaliação
        model.compile({
            optimizer: tf.train.adam(0.01),      // Otimizador Adam com taxa de aprendizado de 0.01
            loss: 'binaryCrossentropy',          // Função de erro para classificação binária (sim/não comprou)
            metrics: ['accuracy']                // Monitora a acurácia durante o treino
        })

        // Inicia o treinamento com os dados fornecidos (xs: entradas, ys: rótulos)
        await model.fit(trainData.xs, trainData.ys, {
            epochs: 100,                         // O modelo verá os dados 100 vezes
            batchSize: 32,                       // Processa 32 exemplos por vez antes de atualizar os pesos
            shuffle: true,                       // Embaralha os dados a cada época para evitar viés de ordem
            callbacks: {
                // Função chamada ao final de cada época para reportar o progresso
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
     * Prepara os tensores de entrada (xs) e saída (ys) para o treinamento.
     */
    createTrainingData(context) {
        const inputs = [] // Armazenará os vetores combinados de usuário e produto
        const labels = [] // Armazenará o resultado real (1 para compra, 0 para não compra)

        // Itera sobre usuários que possuem histórico de compras
        context.users
            .filter(u => u.purchases.length)
            .forEach(user => {
                // Converte o objeto usuário em um vetor numérico
                const userVector = this.encodeUser(user, context).dataSync()

                // Para cada usuário, comparamos com todos os produtos disponíveis
                context.products.forEach(product => {
                    // Transforma o produto em um vetor matemático para que a rede neural possa processá-lo
                    const productsVector = this.encodeProduct(product, context).dataSync()

                    // Verifica se o usuário de fato comprou este produto específico
                    const label = user.purchases.some(
                        purchase => purchase.name == product.name ? 1 : 0
                    )

                    // Combina os dois vetores em um único array de entrada e salva o rótulo
                    inputs.push([...userVector, ...productsVector])
                    labels.push(label)
                })
            });

        // Retorna os dados convertidos em Tensores do TensorFlow
        return {
            xs: tf.tensor2d(inputs),                              // Entradas: [quantidade_exemplos, dimensoes]
            ys: tf.tensor2d(labels, [labels.length, 1]),         // Saídas: [quantidade_exemplos, 1]
            inputDimensions: context.dimensions * 2               // Dimensão total da entrada da rede
        }
    }

    /**
     * Converte um usuário em um vetor representativo baseado no seu histórico.
     */
    encodeUser(user, context) {
        if (user.purchases.length) {
            // Cria um vetor médio de todos os produtos que o usuário já comprou
            return tf.stack(
                user.purchases.map(
                    product => this.encodeProduct(product, context)
                )
            )
                .mean(0) // Tira a média dos vetores para representar o perfil de gosto do usuário
                .reshape([
                    1,
                    context.dimensions
                ])
        }

        // Caso o usuário não possua compras (usuário novo), criamos um perfil padrão (Cold Start)
        // O perfil é baseado inicialmente apenas na idade dele
        return tf.concat1d([
            tf.zeros([1]), // Preço: zero (sem histórico)
            tf.tensor1d([
                normalize(
                    user.age,
                    context.minAge,
                    context.maxAge
                ) * this.WEIGHT.age
            ]), // Idade: normalizada conforme a idade informada
            tf.zeros([context.numCategories]), // Categoria: zero (indefinido)
            tf.zeros([context.numColors])      // Cores: zero (indefinido)
        ]).reshape([
            1,
            context.dimensions
        ])
    }

    /**
     * Transforma as características de um produto (preço, categoria, cor, etc.) em um vetor numérico (Embedding).
     */
    encodeProduct(product, context) {
        // Normaliza o preço para ficar entre 0 e 1 e aplica o peso definido
        const price = tf.tensor1d([
            normalize(
                product.price,
                context.minPrice,
                context.maxPrice
            ) * this.WEIGHT.price
        ])

        // Normaliza a idade média do público do produto e aplica peso
        const age = tf.tensor1d([
            (
                context.productAvgAgeNorm[product.name] ?? 0.5
            ) * this.WEIGHT.age
        ])

        // Converte a categoria em formato One-Hot (vetor binário) com peso
        const category = oneHotWeighted(
            tf,
            context.categoryIndex[product.category],
            context.numCategories,
            this.WEIGHT.category
        )

        // Converte a cor em formato One-Hot com peso
        const color = oneHotWeighted(
            tf,
            context.colorIndex[product.color],
            context.numColors,
            this.WEIGHT.color
        )

        // Concatena todas as características em um único vetor final de características
        return tf.concat1d([price, age, category, color])
    }
}

// ============================================================================
// INSTANCIAÇÃO E ENTRY-POINT (Porta de Entrada do Worker)
// ============================================================================

// Instanciamos o motor de recomendação (a lógica da aplicação)
const engine = new RecommendationEngine();

// Mapeamento de ações que o Worker pode realizar
const handlers = {
    // Ação de treinar o modelo
    [workerEvents.trainModel]: engine.trainModel.bind(engine),
    // Ação de gerar recomendações para um usuário específico
    [workerEvents.recommend]: d => engine.recommend(d.user),
};

// Escuta mensagens enviadas pela thread principal
self.onmessage = e => {
    const { action, ...data } = e.data;
    // Se a ação solicitada existir no nosso mapeamento, nós a executamos
    if (handlers[action]) handlers[action](data);
};

