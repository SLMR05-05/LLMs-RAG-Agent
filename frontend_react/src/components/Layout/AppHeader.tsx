import type { FC } from 'react';
import { Link } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';
import { BookOpen, Plus } from 'lucide-react';

export const AppHeader: FC = () => {
    const activeNotebookId = useAppStore((s) => s.activeNotebookId);
    const notebooks = useAppStore((s) => s.notebooks);

    const activeNotebook = activeNotebookId
        ? notebooks.find((n) => n.id === activeNotebookId || n.notebook_id === activeNotebookId)
        : null;

    const title =
        activeNotebook?.name || 'Technical PRD: University Admissions System';

    return (
        <>
            <header className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-gray-200 bg-white/95 px-4 backdrop-blur-sm flex items-center justify-between gap-3">
                <div className="flex min-w-0 flex-1 items-center gap-3">
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

                <div className="flex flex-shrink-0 items-center gap-2">
                    <button className="px-3 py-2 flex items-center gap-2 bg-black hover:bg-gray-900 text-white rounded-full text-xs font-semibold transition-colors duration-200">
                        <Plus className="w-4 h-4" />
                        Tạo sổ ghi chú
                    </button>
                </div>
            </header>
        </>
    );
};

export default AppHeader;
