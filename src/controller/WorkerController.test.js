import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { WorkerController } from './WorkerController.js';
import { workerEvents } from '../events/constants.js';

describe('WorkerController', () => {
    let mockWorker;
    let mockEvents;
    let controller;

    beforeEach(() => {
        // Mock do Worker
        mockWorker = {
            postMessage: jest.fn(),
            onmessage: null
        };

        // Mock do Event Bus
        mockEvents = {
            onTrainModel: jest.fn(),
            onTrainingComplete: jest.fn(),
            onRecommend: jest.fn(),
            dispatchProgressUpdate: jest.fn(),
            dispatchTrainingComplete: jest.fn(),
            dispatchTFVisorData: jest.fn(),
            dispatchTFVisLogs: jest.fn(),
            dispatchRecommendationsReady: jest.fn()
        };

        controller = new WorkerController({ 
            worker: mockWorker, 
            events: mockEvents 
        });
    });

    it('deve enviar mensagem de inicialização do Supabase corretamente', () => {
        const config = { url: 'http://test.com', key: 'test-key' };
        controller.initSupabase(config);

        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            action: 'init',
            config
        });
    });

    it('deve enviar mensagem de treinamento corretamente', () => {
        const users = [{ id: 1, name: 'Test' }];
        controller.triggerTrain(users);

        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            action: workerEvents.trainModel,
            users
        });
    });

    it('deve enviar mensagem de recomendação corretamente', () => {
        const user = { id: 1 };
        controller.triggerRecommend(user);

        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            action: workerEvents.recommend,
            user
        });
    });

    it('deve registrar os callbacks de eventos na inicialização', () => {
        expect(mockEvents.onTrainModel).toHaveBeenCalled();
        expect(mockEvents.onTrainingComplete).toHaveBeenCalled();
        expect(mockEvents.onRecommend).toHaveBeenCalled();
    });

    it('deve processar mensagens do worker e despachar eventos corretamente', () => {
        // Simula uma mensagem de progresso vinda do worker
        const progressData = { type: workerEvents.progressUpdate, progress: { progress: 50 } };
        mockWorker.onmessage({ data: progressData });

        expect(mockEvents.dispatchProgressUpdate).toHaveBeenCalledWith(progressData.progress);

        // Simula uma mensagem de treinamento concluído
        const completionData = { type: workerEvents.trainingComplete };
        mockWorker.onmessage({ data: completionData });

        expect(mockEvents.dispatchTrainingComplete).toHaveBeenCalledWith(completionData);
    });
});
