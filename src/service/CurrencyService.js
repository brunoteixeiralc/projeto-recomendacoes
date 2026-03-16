export class CurrencyService {
    static #instance;
    #usdToBrlRate = 5.25; // Default fallback rate
    #currentCurrency = 'USD'; // 'USD' or 'BRL'

    constructor() {
        if (CurrencyService.#instance) {
            return CurrencyService.#instance;
        }
        CurrencyService.#instance = this;
    }

    async init() {
        await this.fetchExchangeRate();
    }

    async fetchExchangeRate() {
        try {
            console.log('Fetching exchange rate...');
            const response = await fetch('https://economia.awesomeapi.com.br/json/last/USD-BRL');
            const data = await response.json();
            if (data && data.USDBRL) {
                this.#usdToBrlRate = parseFloat(data.USDBRL.bid);
                console.log(`Exchange rate updated: 1 USD = ${this.#usdToBrlRate} BRL`);
            }
        } catch (error) {
            console.error('Failed to fetch exchange rate, using fallback:', error);
        }
    }

    setCurrency(currency) {
        this.#currentCurrency = currency.toUpperCase();
    }

    convert(value) {
        if (this.#currentCurrency === 'BRL') {
            return value * this.#usdToBrlRate;
        }
        return value;
    }

    format(value) {
        const convertedValue = this.convert(value);
        
        if (this.#currentCurrency === 'BRL') {
            return new Intl.NumberFormat('pt-BR', {
                style: 'currency',
                currency: 'BRL'
            }).format(convertedValue);
        }

        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(convertedValue);
    }
}

export const currencyService = new CurrencyService();
