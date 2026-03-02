import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

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

    debugger

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

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const catalog = await fetch('/data/products.json').then(res => res.json());
    const context = makeContext(users, catalog);

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
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
