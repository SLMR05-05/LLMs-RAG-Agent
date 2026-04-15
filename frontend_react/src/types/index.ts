/**
 * Shared TypeScript types and interfaces for the application
 */

export type PanelAlignment = 'left' | 'center' | 'right';
export type StudioMode = 'outline' | 'faq' | 'briefing' | 'custom';
export type SourceType = 'web' | 'research' | 'file';
export type MessageRole = 'user' | 'assistant';
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface Notebook {
    notebook_id: string;
    name: string;
    created_at: string;
    last_modified: string;
    description?: string;
}

export interface Source {
    id: string;
    title: string;
    type: SourceType;
    url?: string;
    description?: string;
    selected: boolean;
    timestamp?: string;
}

export interface Citation {
    id: number;
    source: Source;
    quote: string;
    page?: number;
}

export interface ChatMessage {
    id: string;
    role: MessageRole;
    content: string;
    html?: string;
    citations?: Citation[];
    timestamp: string;
    isStreaming?: boolean;
}

export interface Job {
    job_id: string;
    status: JobStatus;
    result?: Record<string, unknown>;
    error?: string;
    progress?: number;
}

export interface StudioContent {
    mode: StudioMode;
    title: string;
    content: string;
    lastUpdated: string;
}

export interface AppError {
    code: string;
    message: string;
    details?: Record<string, unknown>;
}
