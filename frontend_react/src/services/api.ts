import type { ChatMessage, ChatSettings, Notebook, SourceDocument } from '../store/useAppStore';

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
    page_count?: number | null;
    file_size_bytes?: number | null;
    created_at?: string | null;
};

export type ApiSourceChunk = {
    chunk_id: string;
    chunk_index: number;
    page_number?: number | null;
    text_content: string;
    created_at?: string | null;
};

export type ApiSourceDetailResponse = {
    notebook_id: string;
    document_id: string;
    source_name: string;
    source_type: string;
    created_at?: string | null;
    page_count?: number | null;
    file_size_bytes?: number | null;
    chunks: ApiSourceChunk[];
    parsed_markdown: string;
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
    chatSettings?: ChatSettings;
};

export type ChatApiResponse = {
    notebook_id: string;
    session_id: string;
    rewritten_question: string;
    answer: string;
    answer_graph: string;
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
        role: 'user' | 'assistant' | 'assistantGraphRag' | 'system';
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

export type WebSourceApiResponse = {
    notebook_id: string;
    job_id: string;
    status: string;
};

export type SourceManageApiResponse = {
    notebook_id: string;
    document_id: string;
    source_name: string;
    source_type: string;
    status: string;
};

export type ChatClearApiResponse = {
    notebook_id: string;
    status: string;
};

function parseSourceType(sourceType: string): SourceDocument['type'] {
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
    async getNotebook(notebookId: string): Promise<Notebook> {
        const data = await request<ApiNotebook>(`/notebooks/${notebookId}`);
        return {
            id: data.notebook_id,
            notebook_id: data.notebook_id,
            name: data.notebook_name,
            created_at: data.created_at || new Date().toISOString(),
            last_modified: data.updated_at || data.created_at || new Date().toISOString(),
        };
    },

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

    async getNotebookSources(notebookId: string): Promise<SourceDocument[]> {
        const data = await request<ApiSourcesResponse>(`/notebooks/${notebookId}/sources`);
        return data.sources.map((source) => ({
            id: source.document_id,
            title: source.source_name,
            type: parseSourceType(source.source_type),
            description: `.${source.source_type}`,
            created_at: source.created_at || undefined,
            page_count: source.page_count ?? null,
            file_size_bytes: source.file_size_bytes ?? null,
            selected: true,
        }));
    },

    async getSourceDetail(notebookId: string, documentId: string): Promise<SourceDocument> {
        const data = await request<ApiSourceDetailResponse>(
            `/notebooks/${notebookId}/sources/${documentId}`
        );

        return {
            id: data.document_id,
            title: data.source_name,
            type: parseSourceType(data.source_type),
            description: `.${data.source_type}`,
            selected: true,
            created_at: data.created_at || undefined,
            page_count: data.page_count ?? null,
            file_size_bytes: data.file_size_bytes ?? null,
            chunks: data.chunks.map((chunk) => ({
                chunk_id: chunk.chunk_id,
                chunk_index: chunk.chunk_index,
                page_number: chunk.page_number ?? null,
                text_content: chunk.text_content,
                created_at: chunk.created_at || undefined,
            })),
            parsed_markdown: data.parsed_markdown,
        };
    },

    getSourcePreviewUrl(notebookId: string, documentId: string): string {
        return `${API_BASE_URL}/notebooks/${notebookId}/sources/${documentId}/preview`;
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
                chatSettings: payload.chatSettings,
            }),
        });
    },

    async clearNotebookChatHistory(notebookId: string): Promise<ChatClearApiResponse> {
        return request<ChatClearApiResponse>(`/notebooks/${notebookId}/chat`, {
            method: 'DELETE',
        });
    },

    async getNotebookChatMessages(
        notebookId: string
    ): Promise<{ sessionId: string | null; messages: ChatMessage[] }> {
        const data = await request<ChatMessagesApiResponse>(`/notebooks/${notebookId}/chat/messages`);
        const uiMessages = data.messages
            .filter(
                (item): item is ChatMessagesApiResponse['messages'][number] & {
                    role: 'user' | 'assistant' | 'assistantGraphRag';
                } => item.role === 'user' || item.role === 'assistant' || item.role === 'assistantGraphRag'
            )
            .map((item) => ({
                id: item.message_id,
                role: item.role,
                content: item.role === 'assistant' || item.role === 'user' ? item.content : undefined,
                contentGraphRag: item.role === 'assistantGraphRag' ? item.content : undefined,
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

    async addNotebookWebLink(notebookId: string, url: string): Promise<WebSourceApiResponse> {
        return request<WebSourceApiResponse>(`/notebooks/${notebookId}/sources/url`, {
            method: 'POST',
            body: JSON.stringify({ url }),
        });
    },

    async renameSource(sourceId: string, sourceName: string): Promise<SourceManageApiResponse> {
        return request<SourceManageApiResponse>(`/sources/${sourceId}`, {
            method: 'PUT',
            body: JSON.stringify({ source_name: sourceName }),
        });
    },

    async deleteSource(sourceId: string): Promise<SourceManageApiResponse> {
        return request<SourceManageApiResponse>(`/sources/${sourceId}`, {
            method: 'DELETE',
        });
    },

    async getJob(jobId: string): Promise<JobApiResponse> {
        return request<JobApiResponse>(`/jobs/${jobId}`);
    },
};

