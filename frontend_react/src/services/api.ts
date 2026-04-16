import type { ChatMessage, Notebook, Source } from '../store/useAppStore';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export type ApiNotebook = {
    notebook_id: string;
    notebook_name: string;
    created_at?: string | null;
    updated_at?: string | null;
};

export type ApiSource = {
    document_id: string;
    source_name: string;
    source_type: string;
    created_at?: string | null;
};

export type ApiSourcesResponse = {
    notebook_id: string;
    total: number;
    sources: ApiSource[];
};

export type ChatPayload = {
    question: string;
    session_id?: string;
    model_name?: string;
    top_k?: number;
    source_names?: string[];
    answer_language?: 'vi' | 'en';
};

export type ChatApiResponse = {
    notebook_id: string;
    session_id: string;
    rewritten_question: string;
    answer: string;
    citations: Array<{
        source_name: string;
        page?: number | null;
        snippet: string;
    }>;
};

export type ChatMessagesApiResponse = {
    notebook_id: string;
    session_id?: string | null;
    messages: Array<{
        message_id: string;
        role: 'user' | 'assistant' | 'system';
        content: string;
        created_at?: string | null;
    }>;
};

export type CitationDetail = ChatApiResponse['citations'][number];

export type JobApiResponse = {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    result?: Record<string, unknown> | null;
    error?: string | null;
};

export type UploadApiResponse = {
    notebook_id: string;
    uploaded_count: number;
    skipped_count: number;
    rejected_count: number;
    rejected: Array<{ filename: string; reason: string }>;
    sources: Array<Record<string, unknown>>;
    job_id?: string | null;
    status?: string | null;
};

function parseSourceType(sourceType: string): Source['type'] {
    const normalized = sourceType.toLowerCase();
    if (normalized.includes('web')) return 'web';
    if (normalized.includes('research')) return 'research';
    return 'file';
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        headers: {
            'Content-Type': 'application/json',
            ...(init?.headers || {}),
        },
        ...init,
    });

    if (!response.ok) {
        let detailMessage = '';

        try {
            const payload = await response.json();
            if (typeof payload?.detail === 'string') {
                detailMessage = payload.detail;
            } else if (typeof payload?.detail?.message === 'string') {
                detailMessage = payload.detail.message;
            } else {
                detailMessage = JSON.stringify(payload);
            }
        } catch {
            detailMessage = await response.text();
        }

        throw new Error(detailMessage || `Request failed with status ${response.status}`);
    }

    return response.json() as Promise<T>;
}

export const apiService = {
    async createNotebook(name: string): Promise<Notebook> {
        const data = await request<ApiNotebook>('/notebooks', {
            method: 'POST',
            body: JSON.stringify({ notebook_name: name }),
        });

        return {
            id: data.notebook_id,
            notebook_id: data.notebook_id,
            name: data.notebook_name,
            created_at: data.created_at || new Date().toISOString(),
            last_modified: data.updated_at || data.created_at || new Date().toISOString(),
        };
    },

    async renameNotebook(notebookId: string, name: string): Promise<Notebook> {
        const data = await request<ApiNotebook>(`/notebooks/${notebookId}`, {
            method: 'PATCH',
            body: JSON.stringify({ notebook_name: name }),
        });

        return {
            id: data.notebook_id,
            notebook_id: data.notebook_id,
            name: data.notebook_name,
            created_at: data.created_at || new Date().toISOString(),
            last_modified: data.updated_at || data.created_at || new Date().toISOString(),
        };
    },

    async deleteNotebook(notebookId: string): Promise<void> {
        const response = await fetch(`${API_BASE_URL}/notebooks/${notebookId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || `Delete failed with status ${response.status}`);
        }
    },

    async getNotebooks(): Promise<Notebook[]> {
        const data = await request<ApiNotebook[]>('/notebooks');
        return data.map((item) => ({
            id: item.notebook_id,
            notebook_id: item.notebook_id,
            name: item.notebook_name,
            created_at: item.created_at || new Date().toISOString(),
            last_modified: item.updated_at || item.created_at || new Date().toISOString(),
        }));
    },

    async getNotebookSources(notebookId: string): Promise<Source[]> {
        const data = await request<ApiSourcesResponse>(`/notebooks/${notebookId}/sources`);
        return data.sources.map((source) => ({
            id: source.document_id,
            title: source.source_name,
            type: parseSourceType(source.source_type),
            description: source.source_type,
            selected: true,
        }));
    },

    async chatNotebook(notebookId: string, payload: ChatPayload): Promise<ChatApiResponse> {
        return request<ChatApiResponse>(`/notebooks/${notebookId}/chat`, {
            method: 'POST',
            body: JSON.stringify({
                question: payload.question,
                session_id: payload.session_id,
                model_name: payload.model_name || 'qwen2.5:1.5b',
                top_k: payload.top_k || 4,
                source_names: payload.source_names,
                answer_language: payload.answer_language,
            }),
        });
    },

    async getNotebookChatMessages(
        notebookId: string
    ): Promise<{ sessionId: string | null; messages: ChatMessage[] }> {
        const data = await request<ChatMessagesApiResponse>(`/notebooks/${notebookId}/chat/messages`);
        const uiMessages = data.messages
            .filter(
                (item): item is ChatMessagesApiResponse['messages'][number] & {
                    role: 'user' | 'assistant';
                } => item.role === 'user' || item.role === 'assistant'
            )
            .map((item) => ({
                id: item.message_id,
                role: item.role,
                content: item.content,
                timestamp: item.created_at || new Date().toISOString(),
            }));

        return {
            sessionId: data.session_id ?? null,
            messages: uiMessages,
        };
    },

    async uploadNotebookSources(notebookId: string, files: File[]): Promise<UploadApiResponse> {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));

        const response = await fetch(`${API_BASE_URL}/notebooks/${notebookId}/sources/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || `Upload failed with status ${response.status}`);
        }

        return response.json() as Promise<UploadApiResponse>;
    },

    async getJob(jobId: string): Promise<JobApiResponse> {
        return request<JobApiResponse>(`/jobs/${jobId}`);
    },
};

