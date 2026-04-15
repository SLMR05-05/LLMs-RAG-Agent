import type { FC } from 'react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Sparkles } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';
import NotebookCard from '../components/Notebook/NotebookCard';
import CreateNotebookModal from '../components/Notebook/CreateNotebookModal';
import RenameNotebookModal from '../components/Notebook/RenameNotebookModal';
import DeleteNotebookModal from '../components/Notebook/DeleteNotebookModal';

const NOTEBOOK_PLACEHOLDERS = Array.from({ length: 8 }, (_, index) => index);

export const HomePage: FC = () => {
    const navigate = useNavigate();
    const notebooks = useAppStore((state) => state.notebooks);
    const activeNotebookId = useAppStore((state) => state.activeNotebookId);
    const isFetchingNotebooks = useAppStore((state) => state.isFetchingNotebooks);
    const fetchNotebooks = useAppStore((state) => state.fetchNotebooks);
    const addNotebook = useAppStore((state) => state.addNotebook);
    const renameNotebook = useAppStore((state) => state.renameNotebook);
    const deleteNotebook = useAppStore((state) => state.deleteNotebook);
    const setActiveNotebook = useAppStore((state) => state.setActiveNotebook);
    const setError = useAppStore((state) => state.setError);

    const [createModalOpen, setCreateModalOpen] = useState(false);
    const [renameModalNotebook, setRenameModalNotebook] = useState<(typeof notebooks)[number] | null>(null);
    const [deleteModalNotebook, setDeleteModalNotebook] = useState<(typeof notebooks)[number] | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        void fetchNotebooks();
    }, [fetchNotebooks]);

    const handleCreateNotebook = async (name: string) => {
        try {
            setIsSubmitting(true);
            setError(null);
            const notebook = await apiService.createNotebook(name);
            addNotebook(notebook);
            setActiveNotebook(notebook.id);
            setCreateModalOpen(false);
            navigate(`/notebook/${notebook.id}`);
        } catch {
            setError('Không thể tạo notebook mới.');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleRenameNotebook = async (name: string) => {
        if (!renameModalNotebook) return;

        try {
            setIsSubmitting(true);
            setError(null);
            const updatedNotebook = await apiService.renameNotebook(renameModalNotebook.id, name);
            renameNotebook(renameModalNotebook.id, updatedNotebook.name);
            setRenameModalNotebook(null);
        } catch {
            setError('Không thể đổi tên notebook.');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDeleteNotebook = async () => {
        if (!deleteModalNotebook) return;

        try {
            setIsSubmitting(true);
            setError(null);
            await apiService.deleteNotebook(deleteModalNotebook.id);
            deleteNotebook(deleteModalNotebook.id);
            if (activeNotebookId === deleteModalNotebook.id) {
                setActiveNotebook(null);
                navigate('/');
            }
            setDeleteModalNotebook(null);
        } catch {
            setError('Không thể xóa notebook.');
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="h-screen overflow-y-auto bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.08),_transparent_35%),linear-gradient(to_bottom_right,_#f9fafb,_#eef2ff)] pt-20">
            <div className="mx-auto max-w-7xl px-6 pb-12 pt-10 lg:px-8">
                <div className="mb-10 flex flex-col gap-6 rounded-3xl border border-gray-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm md:flex-row md:items-center md:justify-between">
                    <div>
                        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
                            <Sparkles className="h-3.5 w-3.5" />
                            Notebook dashboard
                        </div>
                        <h1 className="text-3xl font-semibold tracking-tight text-gray-950 md:text-4xl">
                            Chào mừng trở lại
                        </h1>
                        <p className="mt-3 max-w-2xl text-sm leading-6 text-gray-600 md:text-base">
                            Quản lý notebook, mở nhanh nội dung gần nhất và tạo sổ ghi chú mới chỉ trong một thao tác.
                        </p>
                    </div>

                    <button
                        type="button"
                        onClick={() => setCreateModalOpen(true)}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl bg-black px-5 py-3 text-sm font-semibold text-white transition-all duration-200 hover:-translate-y-0.5 hover:bg-gray-900 hover:shadow-lg"
                    >
                        <Plus className="h-4 w-4" />
                        Tạo mới
                    </button>
                </div>

                <div className="mb-5 flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-semibold text-gray-950">Danh sách sổ ghi chú</h2>
                        <p className="text-sm text-gray-500">Chọn notebook để mở giao diện 3 cột.</p>
                    </div>
                    <div className="text-sm text-gray-500">
                        {notebooks.length} notebook
                    </div>
                </div>

                {isFetchingNotebooks ? (
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
                        {NOTEBOOK_PLACEHOLDERS.map((index) => (
                            <div
                                key={index}
                                className="h-44 animate-pulse rounded-2xl border border-gray-200 bg-white/80 p-5 shadow-sm"
                            >
                                <div className="flex items-start justify-between">
                                    <div className="h-12 w-12 rounded-2xl bg-gray-200" />
                                    <div className="h-8 w-8 rounded-full bg-gray-200" />
                                </div>
                                <div className="mt-6 space-y-3">
                                    <div className="h-4 w-3/4 rounded bg-gray-200" />
                                    <div className="h-4 w-5/6 rounded bg-gray-200" />
                                    <div className="h-3 w-1/2 rounded bg-gray-200" />
                                </div>
                            </div>
                        ))}
                    </div>
                ) : notebooks.length === 0 ? (
                    <div className="flex min-h-[420px] items-center justify-center rounded-3xl border border-dashed border-gray-300 bg-white/70 px-6 py-16 text-center shadow-sm">
                        <div className="max-w-md">
                            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
                                <Sparkles className="h-7 w-7" />
                            </div>
                            <h3 className="text-2xl font-semibold text-gray-950">
                                Hãy tạo sổ ghi chú đầu tiên của bạn
                            </h3>
                            <p className="mt-3 text-sm leading-6 text-gray-600">
                                Bạn chưa có notebook nào. Tạo notebook mới để bắt đầu tải tài liệu, chat với nội dung và quản lý nguồn.
                            </p>
                            <button
                                type="button"
                                onClick={() => setCreateModalOpen(true)}
                                className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-black px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-gray-900"
                            >
                                <Plus className="h-4 w-4" />
                                Tạo notebook mới
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
                        {notebooks.map((notebook) => (
                            <div key={notebook.id} className="animate-fade-in">
                                <NotebookCard
                                    notebook={notebook}
                                    onRename={(nextNotebook) => setRenameModalNotebook(nextNotebook)}
                                    onDelete={(nextNotebook) => setDeleteModalNotebook(nextNotebook)}
                                />
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <CreateNotebookModal
                open={createModalOpen}
                onClose={() => setCreateModalOpen(false)}
                onCreate={handleCreateNotebook}
                isSubmitting={isSubmitting}
            />

            <RenameNotebookModal
                open={renameModalNotebook !== null}
                title="Đổi tên notebook"
                initialName={renameModalNotebook?.name ?? ''}
                onClose={() => setRenameModalNotebook(null)}
                onSubmit={handleRenameNotebook}
                isSubmitting={isSubmitting}
            />

            <DeleteNotebookModal
                open={deleteModalNotebook !== null}
                notebookName={deleteModalNotebook?.name ?? ''}
                onClose={() => setDeleteModalNotebook(null)}
                onConfirm={handleDeleteNotebook}
                isSubmitting={isSubmitting}
            />
        </div>
    );
};

export default HomePage;
