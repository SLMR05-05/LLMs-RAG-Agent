export type SupportedLanguage = 'vi' | 'en';

const VIETNAMESE_KEYWORDS = new Set([
    'toi', 'ban', 'la', 'khong', 'duoc', 'cua', 'trong', 'cho', 'voi',
    'nhung', 'mot', 'nhieu', 'tai', 'sao', 'the', 'nao', 'bao', 'cau',
    'hoi', 'nguon', 'tai', 'lieu', 'vui', 'long', 'giup', 'tom', 'tat',
]);

export function detectQuestionLanguage(question: string): SupportedLanguage {
    const lowered = question.toLowerCase();
    const viChars = 'áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ';

    for (const char of viChars) {
        if (lowered.includes(char)) {
            return 'vi';
        }
    }

    const tokens = lowered
        .split(/\s+/)
        .map((token) => token.replace(/[^a-zA-Z]/g, ''))
        .filter(Boolean);

    const keywordMatches = tokens.reduce((count, token) => {
        return VIETNAMESE_KEYWORDS.has(token) ? count + 1 : count;
    }, 0);

    return keywordMatches >= 2 ? 'vi' : 'en';
}
