import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from '@headlessui/react';
import { Fragment, useEffect, useRef, useState, type FC } from 'react';

interface CreateNotebookModalProps {
    open: boolean;
    onClose: () => void;
    onCreate: (name: string) => Promise<void> | void;
    isSubmitting?: boolean;
}

export const CreateNotebookModal: FC<CreateNotebookModalProps> = ({
    open,
    onClose,
    onCreate,
    isSubmitting = false,
}) => {
    const [name, setName] = useState('');
    const [error, setError] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!open) {
            setName('');
            setError(null);
            return;
        }

        window.setTimeout(() => {
            inputRef.current?.focus();
        }, 0);
    }, [open]);

    const normalize = (value: string) => value.replace(/\s+/g, ' ').trim();

    const validate = (value: string) => {
        const normalized = normalize(value);
        if (normalized.length < 3) return 'Tên notebook phải có ít nhất 3 ký tự.';
        if (normalized.length > 80) return 'Tên notebook không được vượt quá 80 ký tự.';
        return null;
    };

    const handleSubmit = async () => {
        const normalized = normalize(name);
        const nextError = validate(normalized);
        if (nextError) {
            setError(nextError);
            return;
        }

        setError(null);
        await onCreate(normalized);
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter' && name.trim() && !isSubmitting) {
            handleSubmit();
        }
    };

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
                                Tạo notebook mới
                            </DialogTitle>
                            <p className="mt-1 text-sm text-gray-500">
                                Đặt tên để bắt đầu lưu tài liệu và trò chuyện với nội dung.
                            </p>

                            <div className="mt-5">
                                <label className="mb-2 block text-sm font-medium text-gray-700">
                                    Tên notebook
                                </label>
                                <input
                                    ref={inputRef}
                                    value={name}
                                    onChange={(event) => {
                                        setName(event.target.value);
                                        if (error) {
                                            setError(null);
                                        }
                                    }}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Nhập tên notebook"
                                    className={`w-full rounded-2xl border bg-gray-50 px-4 py-3 text-sm text-gray-900 outline-none transition-colors focus:bg-white ${error ? 'border-red-300 focus:border-red-500' : 'border-gray-200 focus:border-blue-500'}`}
                                />
                                {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
                            </div>

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
                                    disabled={!name.trim() || isSubmitting}
                                    onClick={handleSubmit}
                                    className="rounded-2xl bg-black px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-gray-900 disabled:cursor-not-allowed disabled:bg-gray-300"
                                >
                                    {isSubmitting ? 'Đang xử lý...' : 'Tạo notebook'}
                                </button>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </Dialog>
        </Transition>
    );
};

export default CreateNotebookModal;
