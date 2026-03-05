import { makeContext, oneHotWeighted } from './dataProcessor.js';
import { jest } from '@jest/globals';

describe('Data Processor Utils', () => {
    describe('makeContext()', () => {
        it('deve extrair metadados e limites matemáticos corretamente dos arrays de usuários e produtos', () => {
            const users = [
                { age: 20, purchases: [{ name: 'Produto A' }, { name: 'Produto B' }] },
                { age: 30, purchases: [{ name: 'Produto A' }] }
            ];

            const products = [
                { name: 'Produto A', price: 50, color: 'Azul', category: 'Roupas' },
                { name: 'Produto B', price: 100, color: 'Vermelho', category: 'Calçados' },
                { name: 'Produto C', price: 150, color: 'Azul', category: 'Roupas' } // Produto C nunca foi comprado
            ];

            const ctx = makeContext(users, products);

            // Verificando extração de mínimos e máximos globais
            expect(ctx.minAge).toBe(20);
            expect(ctx.maxAge).toBe(30);
            expect(ctx.minPrice).toBe(50);
            expect(ctx.maxPrice).toBe(150);

            // Verificando contagem de Cores e Categorias Únicas
            // Cores únicas: Azul e Vermelho (2)
            expect(ctx.numColors).toBe(2);
            expect(ctx.colorIndex['Azul']).toBeDefined();
            expect(ctx.colorIndex['Vermelho']).toBeDefined();

            // Categorias únicas: Roupas e Calçados (2)
            expect(ctx.numCategories).toBe(2);
            expect(ctx.categoryIndex['Roupas']).toBeDefined();
            expect(ctx.categoryIndex['Calçados']).toBeDefined();

            // Verificando cálculo de Idades Normalizadas
            // Produto A: Compradores têm 20 e 30 anos -> Média: 25. Normalizado entre 20 e 30 = 0.5 centralizado.
            expect(ctx.productAvgAgeNorm['Produto A']).toBe(0.5);
            // Produto B: Comprador tem 20 anos -> Média 20. Normalizado = 0.
            expect(ctx.productAvgAgeNorm['Produto B']).toBe(0);
            // Produto C: Nunca foi vendido -> Recebe a Média do catálogo (20 + 30 / 2) -> Média 25. Normalizado = 0.5.
            expect(ctx.productAvgAgeNorm['Produto C']).toBe(0.5);

            // Dimensions da Rede Neural: (2 default) + (2 cores) + (2 categorias) = 6
            expect(ctx.dimensions).toBe(6);

            // Garante que catálogos foram populados no retorno
            expect(ctx.products).toEqual(products);
            expect(ctx.users).toEqual(users);
        });
    });

    describe('oneHotWeighted()', () => {
        it('deve encadear chamadas às funções do TensorFlow corretamente repassando o peso final', () => {
            // Simulando (Mocking) os métodos aninhados do TensorFlow.js
            // pois neste ambiente não precisamos importar o backend C++ todo do pacote para testar uma lógica
            const mulMock = jest.fn().mockReturnValue('TENSOR_PONDERADO_FINAL');
            const castMock = jest.fn().mockReturnValue({ mul: mulMock });
            const oneHotMock = jest.fn().mockReturnValue({ cast: castMock });

            // Substituindo a injeção do objeto global do Tensor
            const tfMock = { oneHot: oneHotMock };

            const result = oneHotWeighted(tfMock, 1, 3, 0.5);

            // Confirma o encadeamento: tf.oneHot().cast().mul()
            expect(tfMock.oneHot).toHaveBeenCalledWith(1, 3);
            expect(castMock).toHaveBeenCalledWith('float32');
            expect(mulMock).toHaveBeenCalledWith(0.5);

            // Confirma que retorna a variável manipulada da última ponta (multiplicação)
            expect(result).toBe('TENSOR_PONDERADO_FINAL');
        });
    });
});
