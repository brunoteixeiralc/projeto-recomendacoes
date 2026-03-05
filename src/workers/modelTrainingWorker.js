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

        // Cria uma nova propriedade no contexto chamada productVector, mapeando cada produto pra um formato novo
        context.productVector = products.map(product => {
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

    recommend(user) {
        // Se quisermos recomendar, usaremos tranquilamente o escopo isolado desta classe ('this')
        console.log('Will recommend for user:', user, 'using ctx:', this.globalCtx)
        // postMessage({
        //     type: workerEvents.recommend,
        //     user,
        //     recommendations: []
        // });
    }

    // ============================================================================
    // MODELO / INTELIGÊNCIA ARTIFICIAL
    // ============================================================================

    async configureNeuralNetAndTrain(trainData) {
        const model = tf.sequential()

        model.add(tf.layers.dense({
            inputShape: [trainData.inputDimensions],
            units: 128,
            activation: 'relu'
        }))

        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }))

        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }))

        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }))

        model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        })

        await model.fit(trainData.xs, trainData.ys, {
            epochs: 100,
            batchSize: 32,
            shuffle: true,
            callbacks: {
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

    createTrainingData(context) {
        const inputs = []
        const labels = []

        context.users
            .filter(u => u.purchases.length)
            .forEach(user => {
                const userVector = this.encodeUser(user, context).dataSync()

                context.products.forEach(product => {
                    const productVector = this.encodeProduct(product, context).dataSync()

                    const label = user.purchases.some(
                        purchase => purchase.name == product.name ? 1 : 0
                    )

                    inputs.push([...userVector, ...productVector])
                    labels.push(label)
                })
            });

        return {
            xs: tf.tensor2d(inputs),
            ys: tf.tensor2d(labels, [labels.length, 1]),
            inputDimensions: context.dimensions * 2
        }
    }

    encodeUser(user, context) {
        if (user.purchases.length) {
            return tf.stack(
                user.purchases.map(
                    product => this.encodeProduct(product, context)
                )
            )
                .mean(0)
                .reshape([
                    1,
                    context.dimensions
                ])
        }
    }

    encodeProduct(product, context) {
        const price = tf.tensor1d([
            normalize(
                product.price,
                context.minPrice,
                context.maxPrice
            ) * this.WEIGHT.price
        ])

        const age = tf.tensor1d([
            (
                context.productAvgAgeNorm[product.name] ?? 0.5
            ) * this.WEIGHT.age
        ])

        const category = oneHotWeighted(
            tf,
            context.categoryIndex[product.category],
            context.numCategories,
            this.WEIGHT.category
        )

        const color = oneHotWeighted(
            tf,
            context.colorIndex[product.color],
            context.numColors,
            this.WEIGHT.color
        )

        return tf.concat1d([price, age, category, color])
    }
}

// ============================================================================
// INSTANCIAÇÃO E ENTRY-POINT (Porta de Entrada do Worker)
// ============================================================================

// Instanciamos o cérebro! Todo estado (this) ficará isolado aqui de forma segura:
const engine = new RecommendationEngine();

const handlers = {
    // Quando receber mensagem de treino, repassamos para dentro da Classe, amarrando o escopo (bind)
    [workerEvents.trainModel]: engine.trainModel.bind(engine),
    // Quando receber mensagem de recomendação, idem
    [workerEvents.recommend]: d => engine.recommend(d.user),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
