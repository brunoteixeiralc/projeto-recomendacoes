import { workerEvents } from "../events/constants.js";

/**
 * Controller responsável por gerenciar a comunicação entre a thread principal e o Web Worker.
 */
export class WorkerController {
    #worker; // Referência para o Web Worker (TensorFlow)
    #events; // Referência para o barramento de eventos da aplicação
    #alreadyTrained = false; // Flag para controlar se o modelo já está pronto

    constructor({ worker, events }) {
        this.#worker = worker;
        this.#events = events;
        this.#alreadyTrained = false;
        // Inicia a configuração dos escutadores
        this.init();
    }

    /**
     * Inicializa o controller configurando os callbacks
     */
    async init() {
        this.setupCallbacks();
    }

    /**
     * Singleton ou método estático para facilitar a criação da instância
     */
    static init(deps) {
        return new WorkerController(deps);
    }

    /**
     * Configura todos os escutadores de eventos de UI e mensagens vindas do Worker
     */
    setupCallbacks() {
        // Quando a UI pede para treinar o modelo
        this.#events.onTrainModel((data) => {
            this.#alreadyTrained = false; // Reseta o estado enquanto treina
            this.triggerTrain(data); // Manda o comando para o Worker
        });

        // Quando o Worker avisa que terminou o treinamento
        this.#events.onTrainingComplete(() => {
            this.#alreadyTrained = true; // Marca como pronto para recomendações
        });

        // Quando a UI pede uma recomendação para um usuário selecionado
        this.#events.onRecommend((data) => {
            // Só permite recomendar se o treinamento já tiver sido feito
            if (!this.#alreadyTrained) return
            this.triggerRecommend(data); // Manda o comando para o Worker
        });

        // Lista de eventos que não queremos logar no console para não floodar
        const eventsToIgnoreLogs = [
            workerEvents.progressUpdate,
            workerEvents.trainingLog,
            workerEvents.tfVisData,
            workerEvents.tfVisLogs,
            workerEvents.trainingComplete,
        ]

        // Escuta todas as mensagens enviadas pelo Worker via postMessage
        this.#worker.onmessage = (event) => {
            // Loga a mensagem se ela não for uma das ignoradas acima
            if (!eventsToIgnoreLogs.includes(event.data.type))
                console.log('Worker Message:', event.data);

            // Se for atualização de progresso do treinamento
            if (event.data.type === workerEvents.progressUpdate) {
                this.#events.dispatchProgressUpdate(event.data.progress);
            }

            // Se for o aviso de treinamento concluído
            if (event.data.type === workerEvents.trainingComplete) {
                this.#events.dispatchTrainingComplete(event.data);
            }

            // Repassa dados de visualização técnica do TensorFlow (opcional)
            if (event.data.type === workerEvents.tfVisData) {
                this.#events.dispatchTFVisorData(event.data.data);
            }

            // Repassa logs de treinamento (loss/accuracy) para o gráfico
            if (event.data.type === workerEvents.trainingLog) {
                this.#events.dispatchTFVisLogs(event.data);
            }

            // Quando as recomendações chegam ordenadas e prontas
            if (event.data.type === workerEvents.recommend) {
                this.#events.dispatchRecommendationsReady(event.data);
            }
        };
    }

    /**
     * Envia os dados dos usuários para o Worker iniciar o estudo
     */
    triggerTrain(users) {
        this.#worker.postMessage({ action: workerEvents.trainModel, users });
    }

    /**
     * Envia o perfil de um usuário para o Worker gerar a recomendação
     */
    triggerRecommend(user) {
        this.#worker.postMessage({ action: workerEvents.recommend, user });
    }

    /**
     * Configura remotamente o Supabase dentro do Worker passando URL e Chave
     */
    initSupabase(config) {
        this.#worker.postMessage({ action: 'init', config });
    }
}