const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = {
    // Notebooks
    async getNotebooks() {
        const res = await fetch(`${API_BASE_URL}/notebooks`);
        if (!res.ok) throw new Error('Failed to fetch notebooks');
        return res.json();
    },

    async createNotebook(name: string) {
        const res = await fetch(`${API_BASE_URL}/notebooks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        if (!res.ok) throw new Error('Failed to create notebook');
        return res.json();
    },

    // Chat
    async chat(notebookId: string, message: string) {
        const res = await fetch(`${API_BASE_URL}/notebooks/${notebookId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });
        if (!res.ok) throw new Error('Failed to send chat message');
        return res.json();
    },

    // Jobs
    async getJob(jobId: string) {
        const res = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
        if (!res.ok) throw new Error('Failed to fetch job');
        return res.json();
    },

    // Indexing
    async indexNotebook(notebookId: string, sourceIds: string[], chunkSize: number = 256) {
        const res = await fetch(`${API_BASE_URL}/notebooks/${notebookId}/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_ids: sourceIds, chunk_size: chunkSize }),
        });
        if (!res.ok) throw new Error('Failed to index notebook');
        return res.json();
    },
};
