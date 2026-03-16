// Importação dos Controllers, Services e Views da aplicação
import { UserController } from './controller/UserController.js';
import { ProductController } from './controller/ProductController.js';
import { ModelController } from './controller/ModelTrainingController.js';
import { TFVisorController } from './controller/TFVisorController.js';
import { TFVisorView } from './view/TFVisorView.js';
import { UserService } from './service/UserService.js';
import { ProductService } from './service/ProductService.js';
import { UserView } from './view/UserView.js';
import { ProductView } from './view/ProductView.js';
import { ModelView } from './view/ModelTrainingView.js';
import Events from './events/events.js';
import { WorkerController } from './controller/WorkerController.js';

// Importação das configurações globais (URL e Chave do Supabase)
// Nota: O arquivo config.js é ignorado pelo Git para segurança.
import { config } from './config.js';

// Instanciação dos serviços compartilhados (Dados)
const userService = new UserService();
const productService = new ProductService();

// Instanciação das visualizações (Interface)
const userView = new UserView();
const productView = new ProductView();
const modelView = new ModelView();
const tfVisorView = new TFVisorView();

// Inicialização do Web Worker (TensorFlow.js) para rodar processamento pesado em background
const mlWorker = new Worker('/src/workers/modelTrainingWorker.js', { type: 'module' });

// Configuração do Controller do Worker para gerenciar a comunicação
const w = WorkerController.init({
    worker: mlWorker,
    events: Events
});

// Inicializa o Supabase dentro do Worker passando as chaves de forma segura
w.initSupabase({
    url: config.supabaseUrl,
    key: config.supabaseKey
});

// Carrega a base de usuários padrão e dispara o primeiro treinamento automaticamente
const users = await userService.getDefaultUsers();
w.triggerTrain(users);

// Inicialização do Controller do Treinamento (Slider de % e Botão de Treinar)
ModelController.init({
    modelView,
    userService,
    events: Events,
});

// Inicialização do Controller do TFVisor (Gráficos de Loss e Acurácia)
TFVisorController.init({
    tfVisorView,
    events: Events,
});

// Inicialização do Controller de Produtos (Exibição dos Itens no Catálogo)
ProductController.init({
    productView,
    userService,
    productService,
    events: Events,
});

// Inicialização do Controller de Usuários (Painel Lateral de Gerenciamento)
const userController = UserController.init({
    userView,
    userService,
    productService,
    events: Events,
});

// Renderiza o usuário inicial do sistema (Josézin)
userController.renderUsers({
    "id": 99,
    "name": "Josézin da Silva",
    "age": 30,
    "gender": "masculino",
    "purchases": []
});