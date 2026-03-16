import { i18n } from '../service/TranslationService.js';

export class View {
    constructor() {
        this.loadTemplate = this.loadTemplate.bind(this);
    }

    async loadTemplate(templatePath) {
        const response = await fetch(templatePath);
        let template = await response.text();
        
        // Auto-translate templates by looking for t{{key}} pattern
        return this.translateTemplate(template);
    }

    translateTemplate(template) {
        return template.replace(/t{{(.*?)}}/g, (match, key) => {
            return i18n.t(key.trim());
        });
    }

    replaceTemplate(template, data) {
        let result = template;
        for (const [key, value] of Object.entries(data)) {
            result = result.replace(new RegExp(`{{${key}}}`, 'g'), value);
        }
        return result;
    }
}
