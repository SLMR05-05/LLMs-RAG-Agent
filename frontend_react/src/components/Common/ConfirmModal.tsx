import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from '@headlessui/react';
import { Fragment, type FC } from 'react';

interface ConfirmModalProps {
    open: boolean;
    title: string;
    description: string;
    confirmText?: string;
    cancelText?: string;
    isSubmitting?: boolean;
    onConfirm: () => void | Promise<void>;
    onClose: () => void;
}

export const ConfirmModal: FC<ConfirmModalProps> = ({
    open,
    title,
    description,
    confirmText = 'Xóa',
    cancelText = 'Hủy',
    isSubmitting = false,
    onConfirm,
    onClose,
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
                            <DialogTitle className="text-lg font-semibold text-gray-900">{title}</DialogTitle>
                            <p className="mt-2 text-sm leading-6 text-gray-600">{description}</p>

                            <div className="mt-6 flex items-center justify-end gap-3">
                                <button
                                    type="button"
                                    onClick={onClose}
                                    className="rounded-2xl border border-gray-200 px-4 py-2.5 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50"
                                >
                                    {cancelText}
                                </button>
                                <button
                                    type="button"
                                    disabled={isSubmitting}
                                    onClick={() => void onConfirm()}
                                    className="rounded-2xl bg-red-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:bg-red-300"
                                >
                                    {isSubmitting ? 'Đang xóa...' : confirmText}
                                </button>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </Dialog>
        </Transition>
    );
};

export default ConfirmModal;
