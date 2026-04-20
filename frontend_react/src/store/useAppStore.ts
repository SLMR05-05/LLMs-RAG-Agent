import { create } from 'zustand';
import { apiService } from '../services/api';

// ═══════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════

export interface Notebook {
    id: string;
    notebook_id: string;
    name: string;
    created_at: string;
    last_modified: string;
}

export interface Source {
    id: string;
    title: string;
    type: 'web' | 'research' | 'file';
    url?: string;
    description?: string;
    selected: boolean;
}

export interface SourceChunk {
    chunk_id: string;
    chunk_index: number;
    page_number?: number | null;
    text_content: string;
    created_at?: string;
}

export interface SourceDocument extends Source {
    created_at?: string;
    page_count?: number | null;
    file_size_bytes?: number | null;
    chunks?: SourceChunk[];
    parsed_markdown?: string;
}

export interface ChatSettings {
    responseLength: 'short' | 'medium' | 'long';
    roleplay: string;
    mode: 'normal' | 'study_guide' | 'critical_thinking';
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    citations?: number[];
    citationDetails?: Array<{
        source_name: string;
        page?: number | null;
        snippet: string;
    }>;
    timestamp: string;
    html?: string;
}

export interface ActiveCitation {
    sourceId?: string;
    sourceName: string;
    snippet: string;
}

export interface Job {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    result?: Record<string, unknown>;
    error?: string;
}

export interface UploadJob {
    id: string;
    file: File;
    progress: number;
    status: 'pending' | 'uploading' | 'processing' | 'success' | 'error';
    jobId?: string;
}

// ═══════════════════════════════════════════════════════════
// AI RESPONSES DATA
// ═══════════════════════════════════════════════════════════

export const AI_RESPONSES = [
    `Dựa trên tài liệu PRD, hệ thống sử dụng cơ sở dữ liệu quan hệ để lưu trữ thông tin thí sinh và tổ hợp môn. <strong>Schema chính</strong> bao gồm các bảng: <code>thi_sinh</code>, <code>to_hop</code>, <code>diem_mon</code>, và <code>ket_qua_xet_tuyen</code> <sup class="cite-badge">1</sup>.`,

    `Theo tài liệu, quy trình xét tuyển gồm <strong>4 bước chính</strong>: <br/><br/>
   <ul class="chat-list">
     <li>Thu thập điểm thi từ Bộ GD&ĐT qua API</li>
     <li>Tính điểm ưu tiên cho từng tổ hợp</li>
     <li>Xếp hạng theo từng ngành/trường</li>
     <li>Xác nhận kết quả và thông báo thí sinh</li>
   </ul>`,

    `Hệ thống được thiết kế để xử lý tối đa <strong>4.000 thí sinh</strong> đồng thời với thời gian phản hồi dưới <strong>2 giây</strong> cho mỗi truy vấn. Chiến lược tối ưu Skip Optimization giúp giảm tải đáng kể cho cơ sở dữ liệu.`,

    `Về bảo mật, PRD yêu cầu: <br/><br/>
   <ul class="chat-list">
     <li>Mã hoá dữ liệu cá nhân thí sinh theo chuẩn <strong>AES-256</strong></li>
     <li>Xác thực 2 lớp cho tài khoản quản trị viên</li>
     <li>Nhật ký truy cập (audit log) đầy đủ cho mọi thao tác</li>
   </ul>`,
];

// ═══════════════════════════════════════════════════════════
// STORE INTERFACE
// ═══════════════════════════════════════════════════════════

interface AppStore {
    leftPanelCollapsed: boolean;
    rightPanelCollapsed: boolean;
    toggleLeftPanel: () => void;
    toggleRightPanel: () => void;

    notebooks: Notebook[];
    activeNotebookId: string | null;
    setNotebooks: (notebooks: Notebook[]) => void;
    addNotebook: (notebook: Notebook) => void;
    upsertNotebook: (notebook: Notebook) => void;
    getNotebookById: (id: string) => Notebook | null;
    renameNotebook: (id: string, name: string) => void;
    deleteNotebook: (id: string) => void;
    setActiveNotebook: (id: string | null) => void;
    isFetchingNotebooks: boolean;
    fetchNotebooks: () => Promise<void>;

    sources: SourceDocument[];
    setSources: (sources: SourceDocument[]) => void;
    toggleSourceSelection: (sourceId: string) => void;
    selectAllSources: (selected: boolean) => void;
    hasActiveSources: () => boolean;
    sidebarViewMode: 'list' | 'detail';
    selectedSourceForView: SourceDocument | null;
    openSourceDetail: (source: SourceDocument) => void;
    closeSourceDetail: () => void;
    setSelectedSourceDetail: (source: SourceDocument) => void;
    renameSourceLocal: (sourceId: string, sourceName: string) => void;
    deleteSourceLocal: (sourceId: string) => void;

    chatMessages: ChatMessage[];
    chatSessionByNotebook: Record<string, string>;
    chatSettings: ChatSettings;
    activeCitation: ActiveCitation | null;
    addChatMessage: (message: ChatMessage) => void;
    setChatMessages: (messages: ChatMessage[]) => void;
    setChatSessionForNotebook: (notebookId: string, sessionId: string | null) => void;
    clearChat: () => void;
    resetChatHistory: () => void;
    setChatSettings: (settings: ChatSettings) => void;
    setActiveCitation: (citation: ActiveCitation | null) => void;
    clearActiveCitation: () => void;
    clearChatHistory: (notebookId: string) => Promise<void>;
    isTyping: boolean;
    setIsTyping: (typing: boolean) => void;

    jobs: Map<string, Job>;
    addJob: (job: Job) => void;
    updateJob: (jobId: string, job: Job) => void;

    uploadQueue: UploadJob[];
    addUploadJobs: (files: File[]) => void;
    addWebLink: (notebookId: string, url: string) => Promise<string>;
    updateJobProgress: (uploadId: string, progress: number) => void;
    updateJobStatus: (
        uploadId: string,
        status: UploadJob['status'],
        jobId?: string
    ) => void;
    removeUploadJob: (uploadId: string) => void;

    studioMode: 'outline' | 'faq' | 'briefing' | 'custom';
    setStudioMode: (mode: 'outline' | 'faq' | 'briefing' | 'custom') => void;

    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
    error: string | null;
    setError: (error: string | null) => void;
}

// ═══════════════════════════════════════════════════════════
// ZUSTAND STORE
// ═══════════════════════════════════════════════════════════

export const useAppStore = create<AppStore>((set, get) => ({
    leftPanelCollapsed: false,
    rightPanelCollapsed: false,
    toggleLeftPanel: () =>
        set((state: AppStore) => ({ leftPanelCollapsed: !state.leftPanelCollapsed })),
    toggleRightPanel: () =>
        set((state: AppStore) => ({ rightPanelCollapsed: !state.rightPanelCollapsed })),

    notebooks: [],
    activeNotebookId: null,
    setNotebooks: (notebooks: Notebook[]) => set({ notebooks }),
    addNotebook: (notebook: Notebook) =>
        set((state: AppStore) => ({
            notebooks: [notebook, ...state.notebooks],
        })),
    upsertNotebook: (notebook: Notebook) =>
        set((state: AppStore) => {
            const existingIndex = state.notebooks.findIndex(
                (item) => item.id === notebook.id || item.notebook_id === notebook.id
            );

            if (existingIndex < 0) {
                return { notebooks: [notebook, ...state.notebooks] };
            }

            const nextNotebooks = [...state.notebooks];
            nextNotebooks[existingIndex] = notebook;
            return { notebooks: nextNotebooks };
        }),
    getNotebookById: (id: string) => {
        const notebook = get().notebooks.find((item) => item.id === id || item.notebook_id === id);
        return notebook || null;
    },
    renameNotebook: (id: string, name: string) =>
        set((state: AppStore) => ({
            notebooks: state.notebooks.map((notebook) =>
                notebook.id === id || notebook.notebook_id === id
                    ? { ...notebook, name }
                    : notebook
            ),
        })),
    deleteNotebook: (id: string) =>
        set((state: AppStore) => ({
            notebooks: state.notebooks.filter(
                (notebook) => notebook.id !== id && notebook.notebook_id !== id
            ),
        })),
    setActiveNotebook: (id: string | null) => set({ activeNotebookId: id }),
    isFetchingNotebooks: false,
    fetchNotebooks: async () => {
        set({ isFetchingNotebooks: true });
        try {
            const notebooks = await apiService.getNotebooks();
            set({ notebooks });
        } finally {
            set({ isFetchingNotebooks: false });
        }
    },

    sources: [],
    setSources: (sources: SourceDocument[]) => set({ sources }),
    toggleSourceSelection: (sourceId: string) =>
        set((state: AppStore) => ({
            sources: state.sources.map((s: SourceDocument) =>
                s.id === sourceId ? { ...s, selected: !s.selected } : s
            ),
        })),
    selectAllSources: (selected: boolean) =>
        set((state: AppStore) => ({
            sources: state.sources.map((s: SourceDocument) => ({ ...s, selected })),
        })),
    hasActiveSources: () => get().sources.some((source) => source.selected),
    sidebarViewMode: 'list',
    selectedSourceForView: null,
    openSourceDetail: (source: SourceDocument) =>
        set({ sidebarViewMode: 'detail', selectedSourceForView: source }),
    closeSourceDetail: () =>
        set({ sidebarViewMode: 'list', selectedSourceForView: null, activeCitation: null }),
    setSelectedSourceDetail: (source: SourceDocument) =>
        set({ selectedSourceForView: source, sidebarViewMode: 'detail' }),
    renameSourceLocal: (sourceId: string, sourceName: string) =>
        set((state: AppStore) => ({
            sources: state.sources.map((source) =>
                source.id === sourceId ? { ...source, title: sourceName } : source
            ),
            selectedSourceForView:
                state.selectedSourceForView?.id === sourceId
                    ? { ...state.selectedSourceForView, title: sourceName }
                    : state.selectedSourceForView,
        })),
    deleteSourceLocal: (sourceId: string) =>
        set((state: AppStore) => ({
            sources: state.sources.filter((source) => source.id !== sourceId),
            selectedSourceForView:
                state.selectedSourceForView?.id === sourceId ? null : state.selectedSourceForView,
            sidebarViewMode:
                state.selectedSourceForView?.id === sourceId ? 'list' : state.sidebarViewMode,
        })),

    chatMessages: [],
    chatSessionByNotebook: {},
    chatSettings: {
        responseLength: 'medium',
        roleplay: '',
        mode: 'normal',
    },
    activeCitation: null,
    addChatMessage: (message: ChatMessage) =>
        set((state: AppStore) => ({
            chatMessages: [...state.chatMessages, message],
        })),
    setChatMessages: (messages: ChatMessage[]) => set({ chatMessages: messages }),
    setChatSessionForNotebook: (notebookId: string, sessionId: string | null) =>
        set((state: AppStore) => {
            if (!sessionId) {
                const { [notebookId]: _, ...rest } = state.chatSessionByNotebook;
                return { chatSessionByNotebook: rest };
            }

            return {
                chatSessionByNotebook: {
                    ...state.chatSessionByNotebook,
                    [notebookId]: sessionId,
                },
            };
        }),
    clearChat: () => set({ chatMessages: [] }),
    resetChatHistory: () => set({ chatMessages: [] }),
    setChatSettings: (settings: ChatSettings) => set({ chatSettings: settings }),
    setActiveCitation: (citation: ActiveCitation | null) => set({ activeCitation: citation }),
    clearActiveCitation: () => set({ activeCitation: null }),
    clearChatHistory: async (notebookId: string) => {
        await apiService.clearNotebookChatHistory(notebookId);
        set((state: AppStore) => {
            const restSessions = { ...state.chatSessionByNotebook };
            delete restSessions[notebookId];
            return {
                chatMessages: [],
                chatSessionByNotebook: restSessions,
                activeCitation: null,
            };
        });
    },
    isTyping: false,
    setIsTyping: (typing: boolean) => set({ isTyping: typing }),

    jobs: new Map(),
    addJob: (job: Job) =>
        set((state: AppStore) => {
            const newJobs = new Map(state.jobs);
            newJobs.set(job.job_id, job);
            return { jobs: newJobs };
        }),
    updateJob: (jobId: string, job: Job) =>
        set((state: AppStore) => {
            const newJobs = new Map(state.jobs);
            newJobs.set(jobId, job);
            return { jobs: newJobs };
        }),

    uploadQueue: [],
    addUploadJobs: (files: File[]) =>
        set((state: AppStore) => ({
            uploadQueue: [
                ...state.uploadQueue,
                ...files.map((file) => ({
                    id: crypto.randomUUID(),
                    file,
                    progress: 0,
                    status: 'pending' as const,
                })),
            ],
        })),
    addWebLink: async (notebookId: string, url: string) => {
        const result = await apiService.addNotebookWebLink(notebookId, url);
        return result.job_id;
    },
    updateJobProgress: (uploadId: string, progress: number) =>
        set((state: AppStore) => ({
            uploadQueue: state.uploadQueue.map((job) =>
                job.id === uploadId ? { ...job, progress } : job
            ),
        })),
    updateJobStatus: (uploadId: string, status: UploadJob['status'], jobId?: string) =>
        set((state: AppStore) => ({
            uploadQueue: state.uploadQueue.map((job) =>
                job.id === uploadId ? { ...job, status, jobId: jobId ?? job.jobId } : job
            ),
        })),
    removeUploadJob: (uploadId: string) =>
        set((state: AppStore) => ({
            uploadQueue: state.uploadQueue.filter((job) => job.id !== uploadId),
        })),

    studioMode: 'outline',
    setStudioMode: (mode: 'outline' | 'faq' | 'briefing' | 'custom') => set({ studioMode: mode }),

    isLoading: false,
    setIsLoading: (loading: boolean) => set({ isLoading: loading }),
    error: null,
    setError: (error: string | null) => set({ error }),
}));
