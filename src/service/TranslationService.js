export class TranslationService {
    static #instance;
    #currentLanguage = 'pt-br';
    #translations = {};
    #onLanguageChange = [];

    constructor() {
        if (TranslationService.#instance) {
            return TranslationService.#instance;
        }
        TranslationService.#instance = this;
    }

    async init() {
        const savedLang = localStorage.getItem('app_language');
        if (savedLang) {
            this.#currentLanguage = savedLang;
        } else {
            const browserLang = navigator.language.toLowerCase();
            this.#currentLanguage = browserLang.startsWith('en') ? 'en' : 'pt-br';
        }
        await this.loadTranslations(this.#currentLanguage);
    }

    async loadTranslations(lang) {
        try {
            const response = await fetch(`./src/translations/${lang}.json`);
            this.#translations = await response.json();
            this.#currentLanguage = lang;
            localStorage.setItem('app_language', lang);
            this.#notifyListeners();
        } catch (error) {
            console.error(`Failed to load translations for ${lang}:`, error);
        }
    }

    t(key) {
        return this.#translations[key] || key;
    }

    getCurrentLanguage() {
        return this.#currentLanguage;
    }

    async setLanguage(lang) {
        if (lang === this.#currentLanguage) return;
        await this.loadTranslations(lang);
    }

    onLanguageChange(callback) {
        this.#onLanguageChange.push(callback);
    }

    #notifyListeners() {
        this.#onLanguageChange.forEach(callback => callback(this.#currentLanguage));
    }

    /**
     * Translates all elements with data-i18n attribute in the document
     */
    translatePage() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (el.tagName === 'INPUT' && el.type === 'placeholder') {
                el.placeholder = this.t(key);
            } else if (el.tagName === 'OPTION') {
                el.textContent = this.t(key);
            } else {
                // Keep icons if present
                const icon = el.querySelector('i');
                if (icon) {
                    const textNode = Array.from(el.childNodes).find(node => node.nodeType === Node.TEXT_NODE);
                    if (textNode) {
                        textNode.textContent = ' ' + this.t(key);
                    } else {
                        el.appendChild(document.createTextNode(' ' + this.t(key)));
                    }
                } else {
                    el.textContent = this.t(key);
                }
            }
        });
    }
}

export const i18n = new TranslationService();
