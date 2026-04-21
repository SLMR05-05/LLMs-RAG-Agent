import type { ChangeEvent, DragEvent, FC } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { apiService } from '../../services/api';
import {
    AlertCircle,
    CheckCircle2,
    ChevronDown,
    Check,
    Eye,
    FileText,
    FileUp,
    Globe,
    Image,
    Loader2,
    MoreVertical,
    PanelLeft,
    PanelLeftOpen,
    Pencil,
    Search,
    Trash2,
    Upload,
    X,
    Zap,
} from 'lucide-react';
import SourceDetailView from './SourceDetailView';

interface LeftSidebarProps {
    collapsed: boolean;
}

type WebLinkQueueItem = {
    id: string;
    url: string;
    status: 'pending' | 'processing' | 'success' | 'error';
    progress: number;
    jobId?: string;
};

const ACCEPTED_EXTENSIONS = new Set([
    '.pdf',
    '.docx',
    '.txt',
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.gif',
    '.tiff',
    '.tif',
]);

function getFileExtension(fileName: string): string {
    return `.${fileName.split('.').pop()?.toLowerCase() ?? ''}`;
}

function getFileIcon(fileName: string) {
    const extension = getFileExtension(fileName);
    if (['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'].includes(extension)) {
        return <Image className="h-4 w-4 text-blue-600" />;
    }
    if (extension === '.pdf') {
        return <FileText className="h-4 w-4 text-red-500" />;
    }
    if (extension === '.docx' || extension === '.txt') {
        return <FileText className="h-4 w-4 text-slate-600" />;
    }
    return <FileUp className="h-4 w-4 text-slate-500" />;
}

function getStatusTone(status: string): string {
    switch (status) {
        case 'processing':
            return 'text-amber-600 bg-amber-50 border-amber-200';
        case 'success':
            return 'text-emerald-600 bg-emerald-50 border-emerald-200';
        case 'error':
            return 'text-red-600 bg-red-50 border-red-200';
        case 'uploading':
            return 'text-blue-600 bg-blue-50 border-blue-200';
        default:
            return 'text-slate-600 bg-slate-50 border-slate-200';
    }
}

function getStatusLabel(status: string): string {
    switch (status) {
        case 'pending':
            return 'Chờ upload';
        case 'uploading':
            return 'Đang upload';
        case 'processing':
            return 'Đang xử lý';
        case 'success':
            return 'Hoàn tất';
        case 'error':
            return 'Lỗi';
        default:
            return status;
    }
}

export const LeftSidebar: FC<LeftSidebarProps> = ({ collapsed }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const batchInFlightRef = useRef(false);
    const uploadQueueRef = useRef(useAppStore.getState().uploadQueue);
    const webQueueRef = useRef<WebLinkQueueItem[]>([]);
    const [isDragging, setIsDragging] = useState(false);
    const [webUrl, setWebUrl] = useState('');
    const [webLinkQueue, setWebLinkQueue] = useState<WebLinkQueueItem[]>([]);
    const [activeMenuSourceId, setActiveMenuSourceId] = useState<string | null>(null);
    const [renamingSourceId, setRenamingSourceId] = useState<string | null>(null);
    const [renameValue, setRenameValue] = useState('');

    const sources = useAppStore((s) => s.sources);
    const activeCitation = useAppStore((s) => s.activeCitation);
    const sidebarViewMode = useAppStore((s) => s.sidebarViewMode);
    const selectedSourceForView = useAppStore((s) => s.selectedSourceForView);
    const openSourceDetail = useAppStore((s) => s.openSourceDetail);
    const closeSourceDetail = useAppStore((s) => s.closeSourceDetail);
    const uploadQueue = useAppStore((s) => s.uploadQueue);
    const activeNotebookId = useAppStore((s) => s.activeNotebookId);
    const toggleLeftPanel = useAppStore((s) => s.toggleLeftPanel);
    const toggleSourceSelection = useAppStore((s) => s.toggleSourceSelection);
    const selectAllSources = useAppStore((s) => s.selectAllSources);
    const setSources = useAppStore((s) => s.setSources);
    const setSelectedSourceDetail = useAppStore((s) => s.setSelectedSourceDetail);
    const renameSourceLocal = useAppStore((s) => s.renameSourceLocal);
    const deleteSourceLocal = useAppStore((s) => s.deleteSourceLocal);
    const setError = useAppStore((s) => s.setError);
    const addUploadJobs = useAppStore((s) => s.addUploadJobs);
    const addWebLink = useAppStore((s) => s.addWebLink);
    const updateJobProgress = useAppStore((s) => s.updateJobProgress);
    const updateJobStatus = useAppStore((s) => s.updateJobStatus);
    const removeUploadJob = useAppStore((s) => s.removeUploadJob);

    useEffect(() => {
        uploadQueueRef.current = uploadQueue;
    }, [uploadQueue]);

    useEffect(() => {
        webQueueRef.current = webLinkQueue;
    }, [webLinkQueue]);

    const allSelected = sources.every((s) => s.selected);
    const pendingQueue = useMemo(
        () => uploadQueue.filter((job) => job.status === 'pending'),
        [uploadQueue]
    );
    const pollableJobIdsKey = useMemo(
        () =>
            [...new Set(
                uploadQueue
                    .filter((job) => job.jobId && job.status !== 'success' && job.status !== 'error')
                    .map((job) => job.jobId as string)
            )]
                .sort()
                .join('|'),
        [uploadQueue]
    );

    const acceptedFileTypes = '.pdf,.docx,.txt,.png,.jpg,.jpeg,.bmp,.gif,.tiff,.tif';

    const pendingWebJobIdsKey = useMemo(
        () =>
            [...new Set(
                webLinkQueue
                    .filter((item) => item.status === 'processing' && item.jobId)
                    .map((item) => item.jobId as string)
            )]
                .sort()
                .join('|'),
        [webLinkQueue]
    );

    const openFilePicker = () => {
        fileInputRef.current?.click();
    };

    const handleFiles = (files: FileList | File[]) => {
        const fileArray = Array.from(files).filter((file) =>
            ACCEPTED_EXTENSIONS.has(getFileExtension(file.name))
        );

        if (fileArray.length === 0) {
            setError('Chỉ hỗ trợ PDF, DOCX, TXT và ảnh.');
            return;
        }

        setError(null);
        addUploadJobs(fileArray);
    };

    const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            handleFiles(event.target.files);
            event.target.value = '';
        }
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDragging(false);
        handleFiles(event.dataTransfer.files);
    };

    const refreshNotebookSources = async (notebookId: string, uploadedTitle?: string) => {
        // Lưu lại selected state của các sources cũ
        const selectedStateMap = new Map(sources.map(s => [s.id, s.selected]));

        const freshSources = await apiService.getNotebookSources(notebookId);

        // Kết hợp selected state từ cũ + set selected cho source vừa upload
        const mergedSources = freshSources.map(source => ({
            ...source,
            selected: uploadedTitle
                ? source.title === uploadedTitle // Set true cho source vừa upload
                : (selectedStateMap.get(source.id) ?? false) // Giữ selected state cũ, mặc định false
        }));

        setSources(mergedSources);
    };

    const handleRenameSource = async (sourceId: string) => {
        const nextName = renameValue.trim();
        if (!nextName) {
            setError('Tên nguồn không được để trống.');
            return;
        }

        try {
            setError(null);
            await apiService.renameSource(sourceId, nextName);
            renameSourceLocal(sourceId, nextName);
            if (activeNotebookId) {
                await refreshNotebookSources(activeNotebookId);
            }
            setRenamingSourceId(null);
            setRenameValue('');
            setActiveMenuSourceId(null);
        } catch (error) {
            const message =
                error instanceof Error && error.message.trim().length > 0
                    ? error.message
                    : 'Không thể đổi tên nguồn.';
            setError(message);
        }
    };

    const handleDeleteSource = async (sourceId: string, sourceTitle: string) => {
        const confirmed = window.confirm(`Bạn có chắc muốn xóa nguồn \"${sourceTitle}\"?`);
        if (!confirmed) {
            return;
        }

        try {
            setError(null);
            await apiService.deleteSource(sourceId);
            deleteSourceLocal(sourceId);
            closeSourceDetail();
            if (activeNotebookId) {
                await refreshNotebookSources(activeNotebookId);
            }
            setActiveMenuSourceId(null);
        } catch (error) {
            const message =
                error instanceof Error && error.message.trim().length > 0
                    ? error.message
                    : 'Không thể xóa nguồn.';
            setError(message);
        }
    };

    const handleAddWebLink = async () => {
        if (!activeNotebookId) {
            setError('Vui lòng chọn notebook trước khi thêm URL.');
            return;
        }

        const rawUrl = webUrl.trim();
        if (!rawUrl) {
            setError('Vui lòng nhập URL.');
            return;
        }

        let normalizedUrl = rawUrl;
        try {
            if (!/^https?:\/\//i.test(normalizedUrl)) {
                normalizedUrl = `https://${normalizedUrl}`;
            }
            normalizedUrl = new URL(normalizedUrl).toString();
        } catch {
            setError('URL không hợp lệ.');
            return;
        }

        const queueId = crypto.randomUUID();
        setWebLinkQueue((prev) => [
            {
                id: queueId,
                url: normalizedUrl,
                status: 'pending',
                progress: 10,
            },
            ...prev,
        ]);

        try {
            setError(null);
            setWebLinkQueue((prev) =>
                prev.map((item) =>
                    item.id === queueId ? { ...item, status: 'processing', progress: 40 } : item
                )
            );
            const jobId = await addWebLink(activeNotebookId, normalizedUrl);
            setWebLinkQueue((prev) =>
                prev.map((item) =>
                    item.id === queueId ? { ...item, jobId, status: 'processing', progress: 65 } : item
                )
            );
            setWebUrl('');
        } catch (error) {
            const message =
                error instanceof Error && error.message.trim().length > 0
                    ? error.message
                    : 'Không thể thêm URL nguồn.';
            setError(message);
            setWebLinkQueue((prev) =>
                prev.map((item) =>
                    item.id === queueId ? { ...item, status: 'error', progress: 0 } : item
                )
            );
        }
    };

    useEffect(() => {
        if (!activeNotebookId || pendingQueue.length === 0 || batchInFlightRef.current) {
            return;
        }

        batchInFlightRef.current = true;
        const batch = [...pendingQueue];
        batch.forEach((job) => updateJobStatus(job.id, 'uploading'));

        const runUploadBatch = async () => {
            try {
                const response = await apiService.uploadNotebookSources(
                    activeNotebookId,
                    batch.map((job) => job.file)
                );

                const rejectedNames = new Set(response.rejected.map((item) => item.filename));
                const acceptedJobs = batch.filter((job) => !rejectedNames.has(job.file.name));
                const rejectedJobs = batch.filter((job) => rejectedNames.has(job.file.name));

                rejectedJobs.forEach((job) => {
                    updateJobProgress(job.id, 0);
                    updateJobStatus(job.id, 'error');
                });

                if (!response.job_id || acceptedJobs.length === 0) {
                    acceptedJobs.forEach((job) => {
                        updateJobProgress(job.id, 100);
                        updateJobStatus(job.id, 'success');
                        window.setTimeout(() => removeUploadJob(job.id), 3000);
                    });

                    if (acceptedJobs.length > 0) {
                        // Lấy file names từ acceptedJobs, activate chỉ file đầu tiên upload
                        const uploadedFileName = acceptedJobs[0]?.file.name;
                        await refreshNotebookSources(activeNotebookId, uploadedFileName);
                    }
                    return;
                }

                acceptedJobs.forEach((job) => {
                    updateJobProgress(job.id, 35);
                    updateJobStatus(job.id, 'processing', response.job_id as string);
                });
            } catch {
                batch.forEach((job) => {
                    updateJobProgress(job.id, 0);
                    updateJobStatus(job.id, 'error');
                });
                setError('Upload thất bại. Vui lòng thử lại.');
            } finally {
                batchInFlightRef.current = false;
            }
        };

        void runUploadBatch();
    }, [activeNotebookId, pendingQueue, removeUploadJob, setError, setSources, updateJobProgress, updateJobStatus]);

    useEffect(() => {
        if (!activeNotebookId || !pollableJobIdsKey) {
            return;
        }

        const intervalId = window.setInterval(async () => {
            const jobIds = [...new Set(
                uploadQueueRef.current
                    .filter((job) => job.jobId && job.status !== 'success' && job.status !== 'error')
                    .map((job) => job.jobId as string)
            )];

            for (const jobId of jobIds) {
                try {
                    const job = await apiService.getJob(jobId);
                    const matchedJobs = uploadQueueRef.current.filter((item) => item.jobId === jobId);

                    if (job.status === 'running' || job.status === 'pending') {
                        matchedJobs.forEach((item) => {
                            updateJobProgress(item.id, Math.max(item.progress, 65));
                            updateJobStatus(item.id, 'processing', jobId);
                        });
                        continue;
                    }

                    if (job.status === 'completed') {
                        matchedJobs.forEach((item) => {
                            updateJobProgress(item.id, 100);
                            updateJobStatus(item.id, 'success', jobId);
                            window.setTimeout(() => removeUploadJob(item.id), 3000);
                        });

                        // Lấy file names từ matchedJobs, activate chỉ file đầu tiên upload
                        const uploadedFileName = matchedJobs[0]?.file.name;
                        await refreshNotebookSources(activeNotebookId, uploadedFileName);
                        continue;
                    }

                    matchedJobs.forEach((item) => {
                        updateJobProgress(item.id, 0);
                        updateJobStatus(item.id, 'error', jobId);
                    });
                } catch {
                    const matchedJobs = uploadQueueRef.current.filter((item) => item.jobId === jobId);
                    matchedJobs.forEach((item) => {
                        updateJobProgress(item.id, 0);
                        updateJobStatus(item.id, 'error', jobId);
                    });
                }
            }
        }, 1500);

        return () => window.clearInterval(intervalId);
    }, [activeNotebookId, pollableJobIdsKey, removeUploadJob, updateJobProgress, updateJobStatus]);

    useEffect(() => {
        if (!activeNotebookId || !pendingWebJobIdsKey) {
            return;
        }

        const intervalId = window.setInterval(async () => {
            const pendingItems = webQueueRef.current.filter(
                (item) => item.status === 'processing' && item.jobId
            );

            for (const item of pendingItems) {
                if (!item.jobId) continue;

                try {
                    const job = await apiService.getJob(item.jobId);

                    if (job.status === 'running' || job.status === 'pending') {
                        setWebLinkQueue((prev) =>
                            prev.map((it) =>
                                it.id === item.id
                                    ? { ...it, status: 'processing', progress: Math.max(it.progress, 75) }
                                    : it
                            )
                        );
                        continue;
                    }

                    if (job.status === 'completed') {
                        setWebLinkQueue((prev) =>
                            prev.map((it) =>
                                it.id === item.id
                                    ? { ...it, status: 'success', progress: 100 }
                                    : it
                            )
                        );

                        updateJobStatus(item.id, 'success', item.jobId);
                        // Truyền URL hoặc title của web link vừa upload
                        await refreshNotebookSources(activeNotebookId, item.url);
                        window.setTimeout(() => {
                            setWebLinkQueue((prev) => prev.filter((it) => it.id !== item.id));
                            removeUploadJob(item.id);
                        }, 3000);
                        continue;
                    }

                    setWebLinkQueue((prev) =>
                        prev.map((it) =>
                            it.id === item.id ? { ...it, status: 'error', progress: 0 } : it
                        )
                    );
                } catch {
                    setWebLinkQueue((prev) =>
                        prev.map((it) =>
                            it.id === item.id ? { ...it, status: 'error', progress: 0 } : it
                        )
                    );
                }
            }
        }, 1500);

        return () => window.clearInterval(intervalId);
    }, [activeNotebookId, pendingWebJobIdsKey, removeUploadJob, updateJobStatus]);

    useEffect(() => {
        setWebLinkQueue([]);
        setActiveMenuSourceId(null);
        setRenamingSourceId(null);
        setRenameValue('');
    }, [activeNotebookId]);

    useEffect(() => {
        if (!activeCitation || sources.length === 0) {
            return;
        }

        const normalizedCitationName = activeCitation.sourceName.trim().toLowerCase();
        const source =
            sources.find((item) => item.id === activeCitation.sourceId) ||
            sources.find((item) => item.title.trim().toLowerCase() === normalizedCitationName) ||
            sources.find((item) => item.title.toLowerCase().includes(normalizedCitationName)) ||
            sources.find((item) => normalizedCitationName.includes(item.title.toLowerCase()));

        if (source) {
            setSelectedSourceDetail(source);
        }
    }, [activeCitation, setSelectedSourceDetail, sources]);

    return (
        <aside
            className={`flex flex-col h-full min-h-0 bg-white border-r border-gray-200 transition-all duration-300 ease-out overflow-hidden ${collapsed ? 'w-12' : 'w-[300px]'
                }`}
        >
            <div className="flex items-center justify-between flex-shrink-0 px-4 py-3.5 border-b border-gray-200">
                {!collapsed && <h2 className="text-sm font-semibold text-gray-900">Nguồn</h2>}
                <button
                    onClick={toggleLeftPanel}
                    className="ml-auto rounded-lg p-2 transition-colors hover:bg-gray-100"
                    title={collapsed ? 'Mở rộng' : 'Thu gọn'}
                >
                    {collapsed ? (
                        <PanelLeftOpen className="w-4 h-4" />
                    ) : (
                        <PanelLeft className="w-4 h-4" />
                    )}
                </button>
            </div>

            {!collapsed && (
                sidebarViewMode === 'detail' && selectedSourceForView && activeNotebookId ? (
                    <SourceDetailView notebookId={activeNotebookId} source={selectedSourceForView} />
                ) : (
                    <div className="flex min-h-0 flex-1 flex-col animate-fade-in">
                        <div
                            className={`m-4 rounded-2xl border border-dashed p-3 transition-all duration-200 ${isDragging
                                ? 'border-blue-400 bg-blue-50/70'
                                : 'border-gray-200 bg-gray-50/70 hover:border-gray-300 hover:bg-gray-50'
                                }`}
                            onDragEnter={(event) => {
                                event.preventDefault();
                                setIsDragging(true);
                            }}
                            onDragOver={(event) => {
                                event.preventDefault();
                                setIsDragging(true);
                            }}
                            onDragLeave={(event) => {
                                event.preventDefault();
                                setIsDragging(false);
                            }}
                            onDrop={handleDrop}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                multiple
                                accept={acceptedFileTypes}
                                className="hidden"
                                onChange={handleInputChange}
                            />

                            <button
                                type="button"
                                onClick={openFilePicker}
                                className="flex w-full items-center justify-center gap-2 rounded-xl bg-white px-3 py-3 text-sm font-semibold text-gray-900 shadow-sm transition-colors duration-200 hover:bg-gray-100"
                            >
                                <Upload className="h-4 w-4" />
                                Thêm nguồn
                            </button>

                            <p className="mt-2 text-center text-[11px] leading-5 text-gray-500">
                                Kéo thả nhiều file hoặc bấm để chọn .pdf, .docx, .txt, ảnh
                            </p>
                        </div>

                        <div className="px-4 pb-4">
                            <div className="rounded-xl border border-gray-200 bg-white p-2">
                                <div className="relative">
                                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                    <input
                                        type="text"
                                        value={webUrl}
                                        onChange={(event) => setWebUrl(event.target.value)}
                                        placeholder="Dán URL trang web..."
                                        className="w-full rounded-lg border border-gray-200 bg-gray-50 py-2 pl-9 pr-3 text-sm text-gray-900 placeholder-gray-500 transition-colors duration-200 focus:outline-none focus:bg-white focus:border-blue-500"
                                    />
                                </div>
                                <button
                                    type="button"
                                    onClick={handleAddWebLink}
                                    className="mt-2 inline-flex w-full items-center justify-center rounded-lg bg-black px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-gray-900"
                                >
                                    Add Link
                                </button>
                            </div>
                        </div>

                        <div className="px-4 pb-4 flex gap-2 flex-wrap">
                            <button className="flex items-center gap-1 rounded-full bg-gray-200 px-3 py-1 text-xs font-medium text-gray-900 transition-colors duration-200 hover:bg-gray-300">
                                <Globe className="w-3 h-3" />
                                Web
                                <ChevronDown className="w-3 h-3 ml-1" />
                            </button>
                            <button className="flex items-center gap-1 rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-900 transition-colors duration-200 hover:bg-gray-200">
                                <Zap className="w-3 h-3" />
                                Nghiên cứu nhanh
                                <ChevronDown className="w-3 h-3 ml-1" />
                            </button>
                        </div>

                        <div className="px-4 pb-4 border-b border-gray-200">
                            <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-600 hover:text-gray-900">
                                <input
                                    type="checkbox"
                                    checked={allSelected}
                                    onChange={(e) => selectAllSources(e.target.checked)}
                                    className="h-4 w-4 rounded border-gray-300 text-blue-600 accent-blue-600"
                                />
                                Chọn tất cả
                            </label>
                        </div>

                        {(webLinkQueue.length > 0 || uploadQueue.length > 0) && (
                            <div className="border-b border-gray-200 px-4 py-3">
                                <div className="mb-2 flex items-center justify-between">
                                    <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                        Hàng đợi xử lý
                                    </h3>
                                    <span className="text-[11px] text-gray-400">{webLinkQueue.length + uploadQueue.length} tác vụ</span>
                                </div>

                                <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
                                    {webLinkQueue.map((item) => {
                                        const tone = getStatusTone(item.status);
                                        return (
                                            <div key={item.id} className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm">
                                                <div className="flex items-start gap-3">
                                                    <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg bg-gray-50">
                                                        <Globe className="h-4 w-4 text-blue-600" />
                                                    </div>
                                                    <div className="min-w-0 flex-1">
                                                        <div className="flex items-center justify-between gap-2">
                                                            <p className="truncate text-sm font-medium text-gray-900">{item.url}</p>
                                                            <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${tone}`}>
                                                                {item.status === 'processing' && <Loader2 className="h-3 w-3 animate-spin" />}
                                                                {item.status === 'success' && <CheckCircle2 className="h-3 w-3" />}
                                                                {item.status === 'error' && <AlertCircle className="h-3 w-3" />}
                                                                {item.status === 'pending' && <FileUp className="h-3 w-3" />}
                                                                {getStatusLabel(item.status)}
                                                            </span>
                                                        </div>
                                                        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
                                                            <div
                                                                className={`h-1.5 rounded-full transition-all duration-300 ${item.status === 'error'
                                                                    ? 'bg-red-500'
                                                                    : item.status === 'success'
                                                                        ? 'bg-emerald-500'
                                                                        : 'bg-blue-600'
                                                                    }`}
                                                                style={{ width: `${item.progress}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}

                                    {uploadQueue.map((job) => {
                                        const statusTone = getStatusTone(job.status);
                                        return (
                                            <div
                                                key={job.id}
                                                className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm"
                                            >
                                                <div className="flex items-start gap-3">
                                                    <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg bg-gray-50">
                                                        {getFileIcon(job.file.name)}
                                                    </div>

                                                    <div className="min-w-0 flex-1">
                                                        <div className="flex items-center justify-between gap-2">
                                                            <p className="truncate text-sm font-medium text-gray-900">
                                                                {job.file.name}
                                                            </p>
                                                            <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusTone}`}>
                                                                {job.status === 'uploading' && <Loader2 className="h-3 w-3 animate-spin" />}
                                                                {job.status === 'processing' && <Loader2 className="h-3 w-3 animate-spin" />}
                                                                {job.status === 'success' && <CheckCircle2 className="h-3 w-3" />}
                                                                {job.status === 'error' && <AlertCircle className="h-3 w-3" />}
                                                                {job.status === 'pending' && <FileUp className="h-3 w-3" />}
                                                                {getStatusLabel(job.status)}
                                                            </span>
                                                        </div>

                                                        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
                                                            <div
                                                                className={`h-1.5 rounded-full transition-all duration-300 ${job.status === 'error'
                                                                    ? 'bg-red-500'
                                                                    : job.status === 'success'
                                                                        ? 'bg-emerald-500'
                                                                        : job.status === 'processing'
                                                                            ? 'bg-amber-500'
                                                                            : 'bg-blue-600'
                                                                    }`}
                                                                style={{ width: `${job.progress}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        <div className="flex-1 overflow-y-auto">
                            {sources.map((source) => (
                                <div
                                    key={source.id}
                                    className="border-b border-gray-100 px-4 py-3 transition-colors duration-150 hover:bg-gray-50"
                                >
                                    <div className="flex items-start gap-3">
                                        <input
                                            type="checkbox"
                                            checked={source.selected}
                                            onChange={() => toggleSourceSelection(source.id)}
                                            className="mt-1 h-4 w-4 rounded border-gray-300 text-blue-600 accent-blue-600"
                                        />
                                        <div className="min-w-0 flex-1">
                                            {renamingSourceId === source.id ? (
                                                <div className="space-y-2">
                                                    <input
                                                        value={renameValue}
                                                        onChange={(event) => setRenameValue(event.target.value)}
                                                        className="w-full rounded-lg border border-gray-200 bg-white px-2.5 py-1.5 text-sm text-gray-900 outline-none focus:border-blue-500"
                                                        placeholder="Nhập tên mới..."
                                                        autoFocus
                                                    />
                                                    <div className="flex items-center gap-1">
                                                        <button
                                                            type="button"
                                                            onClick={() => void handleRenameSource(source.id)}
                                                            className="inline-flex items-center gap-1 rounded-lg border border-emerald-200 px-2 py-1 text-[11px] font-semibold text-emerald-700 hover:bg-emerald-50"
                                                        >
                                                            <Check className="h-3 w-3" />
                                                            Lưu
                                                        </button>
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                setRenamingSourceId(null);
                                                                setRenameValue('');
                                                            }}
                                                            className="inline-flex items-center gap-1 rounded-lg border border-gray-200 px-2 py-1 text-[11px] font-semibold text-gray-700 hover:bg-gray-100"
                                                        >
                                                            <X className="h-3 w-3" />
                                                            Hủy
                                                        </button>
                                                    </div>
                                                </div>
                                            ) : (
                                                <button
                                                    type="button"
                                                    onClick={(event) => {
                                                        event.preventDefault();
                                                        event.stopPropagation();
                                                        openSourceDetail(source);
                                                    }}
                                                    className="max-w-full truncate text-left text-sm font-medium text-gray-900 hover:text-blue-700"
                                                >
                                                    {source.title}
                                                </button>
                                            )}
                                            {source.description && (
                                                <div className="truncate text-xs text-gray-500">
                                                    {source.description}
                                                </div>
                                            )}
                                            <div className="mt-1 text-xs text-gray-400">
                                                {source.type === 'web' && 'Web'}
                                                {source.type === 'research' && 'Nghiên cứu'}
                                                {source.type === 'file' && 'Tệp'}
                                            </div>

                                            {source.type === 'file' && activeNotebookId && (
                                                <button
                                                    type="button"
                                                    onClick={(event) => {
                                                        event.preventDefault();
                                                        event.stopPropagation();
                                                        openSourceDetail(source);
                                                    }}
                                                    className="mt-2 inline-flex items-center gap-1 rounded-lg border border-gray-200 px-2.5 py-1 text-[11px] font-medium text-gray-700 transition-colors hover:bg-gray-100"
                                                >
                                                    <Eye className="h-3.5 w-3.5" />
                                                    Xem chi tiết
                                                </button>
                                            )}
                                        </div>

                                        <div className="relative">
                                            <button
                                                type="button"
                                                onClick={() =>
                                                    setActiveMenuSourceId((prev) =>
                                                        prev === source.id ? null : source.id
                                                    )
                                                }
                                                className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700"
                                            >
                                                <MoreVertical className="h-4 w-4" />
                                            </button>

                                            {activeMenuSourceId === source.id && (
                                                <div className="absolute right-0 z-10 mt-1 w-32 rounded-lg border border-gray-200 bg-white p-1 shadow-lg">
                                                    <button
                                                        type="button"
                                                        onClick={() => {
                                                            setRenamingSourceId(source.id);
                                                            setRenameValue(source.title);
                                                            setActiveMenuSourceId(null);
                                                        }}
                                                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-100"
                                                    >
                                                        <Pencil className="h-3.5 w-3.5" />
                                                        Đổi tên
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={() => void handleDeleteSource(source.id, source.title)}
                                                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50"
                                                    >
                                                        <Trash2 className="h-3.5 w-3.5" />
                                                        Xóa
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )
            )}
        </aside>
    );
};

export default LeftSidebar;
