import { normalize } from './math.js';

describe('Math Utils', () => {
    describe('normalize()', () => {
        it('deve retornar 0 quando o valor for igual ao minimo', () => {
            expect(normalize(10, 10, 100)).toBe(0);
        });

        it('deve retornar 1 quando o valor for igual ao maximo', () => {
            expect(normalize(100, 10, 100)).toBe(1);
        });

        it('deve retornar 0.5 quando o valor estiver exatamente na metade do intervalo', () => {
            expect(normalize(55, 10, 100)).toBe(0.5);
        });

        it('deve prevenir a divisao por zero retornando 0 se min for igual ao maximo', () => {
            expect(normalize(50, 50, 50)).toBe(0);
        });
    });
});
