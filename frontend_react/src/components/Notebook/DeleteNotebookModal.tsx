import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from '@headlessui/react';
import { Fragment, type FC } from 'react';

interface DeleteNotebookModalProps {
    open: boolean;
    notebookName: string;
    onClose: () => void;
    onConfirm: () => Promise<void> | void;
    isSubmitting?: boolean;
}

export const DeleteNotebookModal: FC<DeleteNotebookModalProps> = ({
    open,
    notebookName,
    onClose,
    onConfirm,
    isSubmitting = false,
}) => {
    return (
        <Transition show={open} as={Fragment}>
            <Dialog as="div" className="relative z-50" onClose={onClose}>
                <TransitionChild
                    as={Fragment}
                    enter="ease-out duration-200"
                    enterFrom="opacity-0"
                    enterTo="opacity-100"
                    leave="ease-in duration-150"
                    leaveFrom="opacity-100"
                    leaveTo="opacity-0"
                >
                    <div className="fixed inset-0 bg-gray-950/45 backdrop-blur-sm" />
                </TransitionChild>

                <div className="fixed inset-0 flex items-center justify-center p-4">
                    <TransitionChild
                        as={Fragment}
                        enter="ease-out duration-200"
                        enterFrom="opacity-0 translate-y-4 scale-95"
                        enterTo="opacity-100 translate-y-0 scale-100"
                        leave="ease-in duration-150"
                        leaveFrom="opacity-100 translate-y-0 scale-100"
                        leaveTo="opacity-0 translate-y-4 scale-95"
                    >
                        <DialogPanel className="w-full max-w-md rounded-3xl bg-white p-6 shadow-2xl ring-1 ring-black/5">
                            <DialogTitle className="text-lg font-semibold text-gray-900">
                                Xóa notebook
                            </DialogTitle>
                            <p className="mt-3 text-sm leading-6 text-gray-600">
                                Bạn có chắc muốn xóa notebook <span className="font-semibold text-gray-900">“{notebookName}”</span>? Hành động này sẽ xóa notebook và dữ liệu liên quan trên máy.
                            </p>

                            <div className="mt-6 flex items-center justify-end gap-3">
                                <button
                                    type="button"
                                    onClick={onClose}
                                    className="rounded-2xl border border-gray-200 px-4 py-2.5 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50"
                                >
                                    Hủy
                                </button>
                                <button
                                    type="button"
                                    onClick={onConfirm}
                                    disabled={isSubmitting}
                                    className="rounded-2xl bg-red-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:bg-red-300"
                                >
                                    {isSubmitting ? 'Đang xóa...' : 'Xóa'}
                                </button>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </Dialog>
        </Transition>
    );
};

export default DeleteNotebookModal;
