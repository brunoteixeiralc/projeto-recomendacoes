import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
import { normalize } from '../utils/math.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIGHT = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

function makeContext(users, catalog) {
    // Cria um array só com as idades de todos os usuários
    const age = users.map(u => u.age)
    // Cria um array só com os preços de todos os produtos do catálogo
    const prices = catalog.map(p => p.price)
    // Encontra a menor idade entre todos os usuários
    const minAge = Math.min(...age)
    // Encontra a maior idade entre todos os usuários
    const maxAge = Math.max(...age)
    // Encontra o menor preço entre todos os produtos
    const minPrice = Math.min(...prices)
    // Encontra o maior preço entre todos os produtos
    const maxPrice = Math.max(...prices)
    // Extrai uma lista única (sem repetições) de todas as cores disponíveis no catálogo
    // O 'Set' remove os duplicados, e o 'Array.from' transforma de volta em uma lista normal
    const colors = Array.from(new Set(catalog.map(p => p.color)))
    // Extrai uma lista única (sem repetições) de todas as categorias do catálogo
    const categories = Array.from(new Set(catalog.map(p => p.category)))

    // Cria um dicionário (objeto) associando cada cor a um número (índice).
    // Exemplo do resultado: { "Azul": 0, "Preto": 1, "Branco": 2 }
    // O TensorFlow.js não entende strings ("Azul"), então precisamos converter textos para números.
    const colorIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    )
    // Cria um dicionário associando cada categoria a um número (índice).
    // Exemplo: { "Eletrônicos": 0, "Vestimenta": 1 }
    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    )
    // Calcula a idade média central, somando a máxima e a mínima e dividindo por 2
    // Esse valor será usado para produtos que nunca foram comprados por ninguém
    const midAge = (maxAge + minAge) / 2
    // Objeto vazio que vai guardar a soma de todas as idades dos usuários que compraram cada produto
    const ageSums = {}
    // Objeto vazio que vai guardar a quantidade de vezes que cada produto foi comprado (para tirar a média depois)
    const ageCounts = {}
    // Percorre cada usuário
    users.forEach(user => {
        // Percorre cada compra desse usuário
        user.purchases.forEach(purchase => {
            // Soma a idade deste usuário ao total de idades do produto comprado
            // Se o produto ainda não existir em 'ageSums', começa com 0 e soma a idade
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age
            // Incrementa o contador de vendas deste produto. 
            // Se ainda não existir em 'ageCounts', começa com 0 e soma 1
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1
        })
    })

    // Cria um objeto final associando CAda produto a uma "idade média normalizada" do seu público
    const productAvgAgeNorm = Object.fromEntries(
        // Percorre todos os produtos do catálogo
        catalog.map(product => {
            // Calcula a média de idade de quem comprou o produto (Soma das Idades / Quantidade de Compras)
            // Se o produto nunca foi comprado (ageCounts for 0 ou falso), usa a idade central ('midAge')
            const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge

            // Retorna um array [Nome do Produto, Valor Normalizado de 0 a 1]
            // A normalização transforma a idade média num valor proporcional onde minAge = 0 e maxAge = 1
            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        catalog,
        // Também retorna a base de usuários recebida na função
        users,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        colorIndex,
        categoryIndex,
        // Quantidade total de cores únicas (útil para criar tensores One-Hot Encoding)
        numColors: colors.length,
        // Quantidade total de categorias únicas
        numCategories: categories.length,
        // Quantidade total de Dimensões que o Modelo de Machine Learning vai receber como Input.
        // A lógica é: 2 dimensões 'básicas' (Idade e Preço)
        // Mais o número de Cores possíveis (ex: 5 cores = 5 dimensões com 1 onde a cor ocorre e 0 nas outras)
        // Mais o número de Categorias possíveis (ex: 10 categorias = 10 dimensões extras)
        dimensions: 2 + colors.length + categories.length
    }
}

const oneHotWeighted = (index, length, weight) =>
    // Cria um array de zeros com um único '1' na posição indicada (One-Hot Encoding).
    // Converte esse array para número decimal (float32) e multiplica pelo peso de importância da característica.
    tf.oneHot(index, length).cast('float32').mul(weight)

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
        context.categoryIndex[product.category],
        context.numCategories,
        WEIGHT.category
    )

    const color = oneHotWeighted(
        context.colorIndex[product.color],
        context.numColors,
        WEIGHT.color
    )

    // tf.concat1d empilha e junta todos os sub-vetores que criamos num único Array longo do Tensor.
    // O resultado disso é a representação integral de "1 produto" em uma lista de números!
    return tf.concat1d([price, age, category, color])
}

async function trainModel({ users }) {
    // Registra no console do navegador que o treinamento do modelo se iniciou e mostra a base de usuários recebida
    console.log('Training model with users:', users)

    // Envia uma mensagem de volta para o resto da aplicação indicando que o processo está na metade (50%)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    // Baixa o arquivo 'products.json' e converte a resposta de JSON para um array Javascript
    const catalog = await fetch('/data/products.json').then(res => res.json());

    // Cria o "contexto" usando a nossa função makeContext() que extrai cálculos matemáticos da base
    // como idades médias, preços mínimos e máximos e os índices pra o One-Hot Encoding
    const context = makeContext(users, catalog);

    // Cria uma nova propriedade no contexto chamada productVector, mapeando cada produto pra um formato novo
    context.productVector = catalog.map(product => {
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

    debugger

    // Salva todo esse contexto calculado para uso global dentro deste worker (útil pra recomendação depois)
    _globalCtx = context

    // Simula o disparo de um evento de log de treinamento, como se uma "Rodada" (Epoch) tivesse terminado
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    // Usa um setTimeout pra simular um processo de finalização de treinamento demorado de 1 segundo
    setTimeout(() => {
        // Envia mensagem indicando 100% de progresso
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        // Sinaliza para a aplicação principal que o treinamento inteiro acabou
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);
}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
