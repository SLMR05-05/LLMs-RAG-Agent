import { useMemo, type FC } from 'react';
import { Menu, MenuButton, MenuItem, MenuItems, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { GraduationCap, Scale, Shield, BookOpen, FileText, Pencil, Trash2, MoreVertical } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import type { Notebook } from '../../store/useAppStore';

interface NotebookCardProps {
    notebook: Notebook;
    onRename: (notebook: Notebook) => void;
    onDelete: (notebook: Notebook) => void;
}

const NOTEBOOK_ICONS = [GraduationCap, Scale, Shield, BookOpen, FileText] as const;

export const NotebookCard: FC<NotebookCardProps> = ({ notebook, onRename, onDelete }) => {
    const navigate = useNavigate();

    const Icon = useMemo(() => {
        const index = notebook.id.charCodeAt(0) % NOTEBOOK_ICONS.length;
        return NOTEBOOK_ICONS[index];
    }, [notebook.id]);

    const createdLabel = new Date(notebook.created_at).toLocaleDateString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
    });

    return (
        <div
            role="button"
            tabIndex={0}
            onClick={() => navigate(`/notebook/${notebook.id}`)}
            onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    navigate(`/notebook/${notebook.id}`);
                }
            }}
            className="group relative flex cursor-pointer flex-col overflow-visible rounded-2xl border border-gray-200 bg-white p-5 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-blue-200 hover:shadow-lg"
        >
            <div className="flex items-start justify-between gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-slate-900 to-slate-700 text-white shadow-sm">
                    <Icon className="h-5 w-5" />
                </div>

                <Menu as="div" className="relative text-left">
                    <MenuButton
                        onClick={(event) => event.stopPropagation()}
                        className="inline-flex items-center justify-center rounded-full p-2 text-gray-400 transition-all hover:bg-gray-100 hover:text-gray-800"
                        aria-label="Mở menu notebook"
                    >
                        <MoreVertical className="h-4 w-4" />
                    </MenuButton>

                    <Transition
                        as={Fragment}
                        enter="transition ease-out duration-120"
                        enterFrom="opacity-0 translate-y-1 scale-95"
                        enterTo="opacity-100 translate-y-0 scale-100"
                        leave="transition ease-in duration-100"
                        leaveFrom="opacity-100 translate-y-0 scale-100"
                        leaveTo="opacity-0 translate-y-1 scale-95"
                    >
                        <MenuItems 
                        anchor="bottom end"
                        className="absolute right-0 top-11 z-30 w-52 origin-top-right overflow-hidden rounded-2xl border border-gray-200 bg-white p-1 shadow-[0_18px_50px_rgba(15,23,42,0.14)] ring-1 ring-black/5 focus:outline-none">
                            <div className="px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-gray-400">
                                Notebook actions
                            </div>

                            <MenuItem>
                                {({ focus }) => (
                                    <button
                                        type="button"
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            onRename(notebook);
                                        }}
                                        className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${focus ? 'bg-blue-50 text-blue-700' : 'text-gray-700'}`}
                                    >
                                        <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-gray-100 text-gray-600">
                                            <Pencil className="h-4 w-4" />
                                        </span>
                                        <span className="flex-1 text-left">Đổi tên</span>
                                    </button>
                                )}
                            </MenuItem>

                            <MenuItem>
                                {({ focus }) => (
                                    <button
                                        type="button"
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            onDelete(notebook);
                                        }}
                                        className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${focus ? 'bg-red-50 text-red-700' : 'text-red-600'}`}
                                    >
                                        <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-red-50 text-red-600">
                                            <Trash2 className="h-4 w-4" />
                                        </span>
                                        <span className="flex-1 text-left">Xóa</span>
                                    </button>
                                )}
                            </MenuItem>
                        </MenuItems>
                    </Transition>
                </Menu>
            </div>

            <div className="mt-5">
                <h3 className="line-clamp-2 text-base font-semibold text-gray-900 transition-colors group-hover:text-blue-700">
                    {notebook.name}
                </h3>
                <p className="mt-2 text-sm text-gray-500">Ngày tạo: {createdLabel}</p>
            </div>
        </div>
    );
};

export default NotebookCard;
