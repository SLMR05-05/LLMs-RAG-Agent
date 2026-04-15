import type { FC } from 'react';
import { Link } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';
import { Plus, Share2, Settings, Grid3x3, BookOpen } from 'lucide-react';

export const AppHeader: FC = () => {
    const activeNotebookId = useAppStore((s) => s.activeNotebookId);
    const notebooks = useAppStore((s) => s.notebooks);

    const activeNotebook = activeNotebookId
        ? notebooks.find((n) => n.id === activeNotebookId || n.notebook_id === activeNotebookId)
        : null;

    const title =
        activeNotebook?.name || 'Technical PRD: University Admissions System';

    return (
        <header className="fixed top-0 left-0 right-0 z-50 h-14 bg-white/95 backdrop-blur-sm border-b border-gray-200 flex items-center justify-between px-4 gap-3">
            {/* LEFT: Logo and Title */}
            <div className="flex items-center gap-3 flex-1 min-w-0">
                <Link
                    to="/"
                    className="flex-shrink-0 w-7 h-7 bg-black rounded text-white flex items-center justify-center hover:opacity-80 transition-all duration-200"
                    title="Home"
                >
                    <BookOpen className="w-4 h-4" />
                </Link>
                <h1 className="text-sm font-medium text-gray-900 truncate" title={title}>
                    {title}
                </h1>
            </div>

            {/* RIGHT: Action Buttons */}
            <div className="flex items-center gap-2 flex-shrink-0">
                <button className="px-3 py-2 flex items-center gap-2 bg-black hover:bg-gray-900 text-white rounded-full text-xs font-semibold transition-colors duration-200">
                    <Plus className="w-4 h-4" />
                    Tạo sổ ghi chú
                </button>

                <button className="px-3 py-2 flex items-center gap-2 bg-white hover:bg-gray-100 border border-gray-200 text-gray-900 rounded-full text-xs font-semibold transition-colors duration-200">
                    <Share2 className="w-4 h-4" />
                    Chia sẻ
                </button>

                <div className="w-px h-6 bg-gray-200" />

                <button
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Cài đặt"
                >
                    <Settings className="w-5 h-5 text-gray-600" />
                </button>

                <button
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Ứng dụng"
                >
                    <Grid3x3 className="w-5 h-5 text-gray-600" />
                </button>

                <div className="w-px h-6 bg-gray-200" />

                <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-semibold cursor-pointer hover:bg-blue-600 transition-colors">
                    An
                </div>
            </div>
        </header>
    );
};

export default AppHeader;
