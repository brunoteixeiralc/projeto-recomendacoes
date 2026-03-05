import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
import { normalize } from '../utils/math.js';
import { makeContext, oneHotWeighted } from '../utils/dataProcessor.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = {};

const WEIGHT = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

// ============================================================================
// 1. ENTRY-POINT (Porta de Entrada do Worker)
// ============================================================================

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};

// ============================================================================
// 2. ORQUESTRADORES PRINCIPAIS
// ============================================================================

async function trainModel({ users }) {
    // Registra no console do navegador que o treinamento do modelo se iniciou e mostra a base de usuários recebida
    console.log('Training model with users:', users)

    // Envia uma mensagem de volta para o resto da aplicação indicando que o processo está na metade (50%)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    // Baixa o arquivo 'products.json' e converte a resposta de JSON para um array Javascript
    const products = await fetch('/data/products.json').then(res => res.json());

    // Cria o "contexto" usando a nossa função makeContext() que extrai cálculos matemáticos da base
    // como idades médias, preços mínimos e máximos e os índices pra o One-Hot Encoding
    const context = makeContext(users, products);

    // Cria uma nova propriedade no contexto chamada productVector, mapeando cada produto pra um formato novo
    context.productVector = products.map(product => {
        return {
            // Nome do produto
            name: product.name,
            // Cópia de todos os dados originais desse produto (meta dados)
            meta: { ...product },
            // A representação matemática (vetor) deste produto computada pela função encodeProduct
            // O .dataSync() pede ao TensorFlow para descer a sua variável tipo Tensor e
            // devolver os números de forma síncrona em um Array simples do Javascript.
            // 
            // Exemplo do que será retornado:
            // [
            //   // 1º Posição (PREÇO): Normalizado de 0 a 1 e multiplicado pelo peso de importância (0.2)
            //   0.10,   // (ex: 55 reais está bem no meio entre 10 e 100, x 0.2 de Peso da feature "preço")
            //   
            //   // 2º Posição (IDADE): Normalizada de 0 a 1 e multiplicada pelo peso de importância (0.1)
            //   0.05,   // (ex: 34 anos está na metade entre 18 e 50 anos, x 0.1 de Peso da feature "idade")
            // 
            //   // 3º, 4º e 5º Posição (CATEGORIA): One-Hot Encoding das Categorias ['Roupas', 'Eletrônicos'].
            //   // Como é Roupa, a primeira casa ganha o Peso (0.4) e o resto ganha Zero.
            //   0.4, 
            //   0.0, 
            // 
            //   // 6º, 7º e 8º Posição (COR): One-Hot Encoding das Cores ['Azul', 'Vermelho', 'Preto']
            //   // Como é Vermelho, a cor do MEIO ganha o Peso (0.3) e as outras ganham Zero.
            //   0.0, 
            //   0.3, 
            //   0.0
            // ]
            vector: encodeProduct(product, context).dataSync()
        }
    })

    // Salva todo esse contexto calculado para uso global dentro deste worker (útil pra recomendação depois)
    _globalCtx = context

    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

    // Envia mensagem indicando 100% de progresso
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    // Sinaliza para a aplicação principal que o treinamento inteiro acabou
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}

// ============================================================================
// 3. MODELO / INTELIGÊNCIA ARTIFICIAL
// ============================================================================

async function configureNeuralNetAndTrain(trainData) {
    // 1. Inicializa o modelo da Rede Neural.
    // 'sequential' significa que as camadas da rede serão empilhadas uma após a outra (como uma linha de montagem).
    // A informação vai entrar pela primeira camada, ser processada e repassada para a próxima.
    const model = tf.sequential()

    // 2. Adiciona a PRIMEIRA camada oculta e, implicitamente, a camada de "Entrada" dos dados
    model.add(
        tf.layers.dense({
            // 'inputShape': Define exatamente o tamanho da nossa "Pergunta" (matriz com o Usuário + Produto).
            // O TensorFlow precisa saber com antecedência quantas "features" (colunas) vai receber da base.
            inputShape: [trainData.inputDimensions],
            // 'units': Quantidade de "Neurônios" artificiais nesta camada (128). 
            // Quanto mais neurônios, mais padrões complexos ela consegue memorizar (porém mais lenta fica).
            units: 128,
            // 'activation': 'relu' (Rectified Linear Unit) zera valores negativos e deixa passar os positivos.
            // Ajuda a rede a focar apenas nas características que importam e a cortar o "ruído matemático".
            activation: 'relu'
        })
    )

    // 3. Adiciona a SEGUNDA camada (Camadas Densas / Hidden Layers)
    // Recebe o processamento dos 128 neurônios anteriores e tenta encontrar padrões mais profundos.
    model.add(
        tf.layers.dense({
            // 'units': Vai afunilando a informação, condensando de 128 neurônios para 64.
            units: 64,
            activation: 'relu'
        })
    )

    // 4. Adiciona a TERCEIRA camada
    // Afunila mais ainda a informação, condensando agora para 32 neurônios.
    // Esse processo de afunilamento força a rede a extrair apenas o conhecimento puro que mais importa.
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )

    // 5. Adiciona a QUARTA e ÚLTIMA camada (Camada de Saída / Output Layer)
    // É esta camada que vai dar o "veredicto final" sobre a propensão de compra.
    model.add(
        tf.layers.dense({
            // 'units': Apenas 1 neurônio de saída. Porque nós só queremos 1 única resposta final (Sim ou Não).
            units: 1,
            // 'activation': 'sigmoid'. Essa é a grande jogada matemática da camada de saída final:
            // A curva 'sigmoid' esmaga o resultado da rede para caber ESPECIFICAMENTE entre 0.0 e 1.0.
            // Logo, se a saída for 0.99, significa 99% de chance de recomendação. Se for 0.01 = 1%.
            activation: 'sigmoid'
        })
    )

    // 6. Prepara ("Compila") o modelo acabado de montar, informando como ele deve aprender com os erros.
    model.compile({
        // 'optimizer': O 'adam' é o algoritmo escolhido para ajustar os pesos ("sinapses") após cada erro. 
        // 0.01 é a taxa de aprendizado (learning rate). Um valor equilibrado para não aprender irresponsávelmente rápido e nem muito devagar.
        optimizer: tf.train.adam(0.01),
        // 'loss': A função para calcular a "dor" de quanto o modelo errou o gabarito. 
        // 'binaryCrossentropy' é a métrica padrão ouro quando o nosso objetivo final é apenas "Sim" vs "Não".
        loss: 'binaryCrossentropy',
        // 'metrics': O que nós queremos exibir na tela para acompanhar o treinamento dele (percentual de acertos).
        metrics: ['accuracy']
    })

    // 7. Finalmente, chuta a bola pro gol e INICIA O TREINAMENTO usando nossas matrizes geradas.
    // 'await': O treinamento leva segundos ou minutos, o processador vai ficar bloqueado e precisamos usar Promises.
    await model.fit(trainData.xs, trainData.ys, {
        // 'epochs': Quantas vezes a rede vai iterar pela BASE INTEIRA do zero pra tentar melhorar o aprendizado (100 vezes)
        epochs: 100,
        // 'batchSize': A rede estuda "lotes" de 32 cruzamentos(usuário+produto) por vez, ajusta os pesos, e parte pro próximo lote.
        batchSize: 32,
        // 'shuffle': Muito importante! Embaralha a ordem da nossa tabela a cada 'Epoch', pro modelo não "decorar" e viciar a ordem das respostas.
        shuffle: true,
        // 'callbacks': Funções bônus para disparar a cada etapa do treinamento:
        callbacks: {
            // Ao final de cada 'Epoch' (quando terminar as 100 rodadas na base)...
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

    // (Atenção: A instrução return model estava comentada/ausente no seu código prévio, mantive omitido fielmente,
    //  mas o comum seria ter um 'return model' aqui para que a variável fora da função o receba corretamente!)
}

// ============================================================================
// 4. FUNÇÕES AUXILIARES / PREPARAÇÃO DE DADOS MATEMÁTICOS
// ============================================================================

function createTrainingData(context) {
    // Arrays para guardar as "perguntas" (features/inputs) e as "respostas corretas" (labels/resultados) do nosso treinamento
    const inputs = []
    const labels = []

    // Para cada usuário na base de dados...
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            // Extrai o vetor matemático que representa o perfil deste usuário (baseado no que ele comprou no passado)
            // O '.dataSync()' transforma o "Tensor" do TensorFlow em um simples array de números do Javascript
            const userVector = encodeUser(user, context).dataSync()

            // Agora cruzamos o perfil deste usuário com TODOS os produtos disponíveis no catálogo da loja...
            context.products.forEach(product => {
                // Pega o vetor matemático que representa as características exclusivas deste produto
                const productVector = encodeProduct(product, context).dataSync()

                // Verifica se este usuário comprou ou não este produto analisando o seu histórico.
                // O '.some()' percorre as compras do usuário e retorna 'true' (sim) ou 'false' (não).
                const label = user.purchases.some(
                    purchase => purchase.name == product.name ? 1 : 0
                )

                // Junta o array do perfil do usuário e o array do produto num único array (concatenação).
                // Isso representa o cenário "Este Usuário + Este Produto" (o que a rede neural fará a análise).
                inputs.push([...userVector, ...productVector])

                // Guarda a resposta real se houve compra (true) ou não (false), para a rede neural poder aprender.
                labels.push(label)
            })
        });

    // Ao final, criamos e retornamos os Tensores Bidimensionais do TensorFlow com as matrizes finais prontas
    return {
        // 'xs' simboliza o eixo X ou as Features: a nossa grande tabela com as junções de Usuário + Produto
        xs: tf.tensor2d(inputs),

        // 'ys' simboliza o eixo Y ou os Labels (Classificação): as respostas corretas de true/false para cada junção de cima.
        // Convertidos num formato de coluna vertical: [num_linhas, 1_coluna]
        ys: tf.tensor2d(labels, [labels.length, 1]),

        // Guarda o número exato de dimensões (features) que estamos enviando pro modelo a cada "linha"
        // Como juntamos 1 vetor de Usuário e 1 vetor de Produto (que têm os mesmos tamanhos), multiplicamos por 2
        inputDimensions: context.dimensions * 2
    }
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimensions
            ])
    }
}

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHT.price
    ])

    const age = tf.tensor1d([
        (
            // Busca a idade média (normalizada de 0 a 1) do público que compra este produto.
            // Se o produto nunca foi vendido, recebe um valor neutro padrão de 0.5 (idade média do catálogo).
            // O valor final é multiplicado pelo peso de importância ('WEIGHT.age') que essa métrica tem.
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHT.age
    ])

    const category = oneHotWeighted(
        tf,
        context.categoryIndex[product.category],
        context.numCategories,
        WEIGHT.category
    )

    const color = oneHotWeighted(
        tf,
        context.colorIndex[product.color],
        context.numColors,
        WEIGHT.color
    )

    // tf.concat1d empilha e junta todos os sub-vetores que criamos num único Array longo do Tensor.
    // O resultado disso é a representação integral de "1 produto" em uma lista de números!
    return tf.concat1d([price, age, category, color])
}


