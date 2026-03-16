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
import { i18n } from './service/TranslationService.js';

// Importação das configurações globais (URL e Chave do Supabase)
import { config } from './config.js';

// 1. Inicialização do Sistema de Internacionalização (i18n)
await i18n.init();

// Escuta a mudança de idioma para atualizar a página
const languageSelect = document.querySelector('#languageSelect');
languageSelect.value = i18n.getCurrentLanguage();

languageSelect.addEventListener('change', async (e) => {
    await i18n.setLanguage(e.target.value);
    // Recarrega a página ou re-renderiza componentes principais
    window.location.reload(); 
});

// Traduz os elementos estáticos iniciais do index.html
i18n.translatePage();

// Instanciação dos serviços compartilhados (Dados)
const userService = new UserService();
const productService = new ProductService();

// Instanciação das visualizações (Interface)
const userView = new UserView();
const productView = new ProductView();
const modelView = new ModelView();
const tfVisorView = new TFVisorView();

// Inicialização do Web Worker (TensorFlow.js)
const mlWorker = new Worker('/src/workers/modelTrainingWorker.js', { type: 'module' });

// Configuração do Controller do Worker
const w = WorkerController.init({
    worker: mlWorker,
    events: Events
});

// Inicializa o Supabase no Worker
w.initSupabase({
    url: config.supabaseUrl,
    key: config.supabaseKey
});

// Carrega a base de usuários padrão e dispara o primeiro treinamento
const users = await userService.getDefaultUsers();
w.triggerTrain(users);

// Inicialização do Controller do Treinamento
ModelController.init({
    modelView,
    userService,
    events: Events,
});

// Inicialização do Controller do TFVisor
TFVisorController.init({
    tfVisorView,
    events: Events,
});

// Inicialização do Controller de Produtos
ProductController.init({
    productView,
    userService,
    productService,
    events: Events,
});

// Inicialização do Controller de Usuários
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