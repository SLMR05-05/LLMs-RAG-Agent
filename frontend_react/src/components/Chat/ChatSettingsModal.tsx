import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from '@headlessui/react';
import type { FC } from 'react';
import { Fragment } from 'react';
import { useAppStore, type ChatSettings } from '../../store/useAppStore';

interface ChatSettingsModalProps {
    open: boolean;
    onClose: () => void;
}

const RESPONSE_LENGTH_OPTIONS: Array<{ value: ChatSettings['responseLength']; label: string; description: string }> = [
    { value: 'short', label: 'Ngắn gọn', description: 'Câu trả lời dưới 3 câu' },
    { value: 'medium', label: 'Chi tiết', description: 'Cân bằng giữa ngắn và đầy đủ' },
    { value: 'long', label: 'Phân tích sâu', description: 'Trả lời nhiều lớp, có lập luận' },
];

const MODE_OPTIONS: Array<{ value: ChatSettings['mode']; label: string; description: string }> = [
    { value: 'normal', label: 'Bình thường', description: 'Trả lời trung tính, rõ ràng' },
    { value: 'study_guide', label: 'Hướng dẫn học tập', description: 'Ưu tiên giải thích và ghi nhớ' },
    { value: 'critical_thinking', label: 'Phản biện', description: 'Ưu tiên phân tích và chất vấn giả định' },
];

export const ChatSettingsModal: FC<ChatSettingsModalProps> = ({ open, onClose }) => {
    const chatSettings = useAppStore((state) => state.chatSettings);
    const setChatSettings = useAppStore((state) => state.setChatSettings);

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
                        <DialogPanel className="w-full max-w-2xl rounded-3xl bg-white p-6 shadow-2xl ring-1 ring-black/5">
                            <DialogTitle className="text-lg font-semibold text-gray-900">
                                Cài đặt Chat
                            </DialogTitle>
                            <p className="mt-1 text-sm text-gray-500">
                                Tùy chỉnh phong cách phản hồi của Local RAG theo từng notebook.
                            </p>

                            <div className="mt-6 space-y-6">
                                <section>
                                    <div className="mb-3">
                                        <h3 className="text-sm font-semibold text-gray-900">Độ dài câu trả lời</h3>
                                        <p className="text-xs text-gray-500">Chọn mức chi tiết mong muốn cho phản hồi.</p>
                                    </div>
                                    <div className="grid gap-3 md:grid-cols-3">
                                        {RESPONSE_LENGTH_OPTIONS.map((option) => {
                                            const selected = chatSettings.responseLength === option.value;
                                            return (
                                                <button
                                                    key={option.value}
                                                    type="button"
                                                    onClick={() =>
                                                        setChatSettings({
                                                            ...chatSettings,
                                                            responseLength: option.value,
                                                        })
                                                    }
                                                    className={`rounded-2xl border px-4 py-3 text-left transition-all ${selected ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-200' : 'border-gray-200 bg-white hover:bg-gray-50'}`}
                                                >
                                                    <div className="text-sm font-semibold text-gray-900">{option.label}</div>
                                                    <div className="mt-1 text-xs leading-5 text-gray-500">{option.description}</div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </section>

                                <section>
                                    <label className="mb-2 block text-sm font-semibold text-gray-900">
                                        Nhập vai / Phong cách
                                    </label>
                                    <input
                                        value={chatSettings.roleplay}
                                        onChange={(event) =>
                                            setChatSettings({
                                                ...chatSettings,
                                                roleplay: event.target.value,
                                            })
                                        }
                                        placeholder="Ví dụ: Giáo sư nghiêm khắc, Chuyên gia tóm tắt"
                                        className="w-full rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-900 outline-none transition-colors placeholder:text-gray-400 focus:border-blue-500 focus:bg-white"
                                    />
                                    <p className="mt-2 text-xs text-gray-500">
                                        Để trống nếu muốn dùng phong cách mặc định.
                                    </p>
                                </section>

                                <section>
                                    <div className="mb-3">
                                        <h3 className="text-sm font-semibold text-gray-900">Chế độ chat</h3>
                                        <p className="text-xs text-gray-500">Điều hướng cách AI tổ chức và phân tích câu trả lời.</p>
                                    </div>
                                    <div className="grid gap-3 md:grid-cols-3">
                                        {MODE_OPTIONS.map((option) => {
                                            const selected = chatSettings.mode === option.value;
                                            return (
                                                <button
                                                    key={option.value}
                                                    type="button"
                                                    onClick={() =>
                                                        setChatSettings({
                                                            ...chatSettings,
                                                            mode: option.value,
                                                        })
                                                    }
                                                    className={`rounded-2xl border px-4 py-3 text-left transition-all ${selected ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-200' : 'border-gray-200 bg-white hover:bg-gray-50'}`}
                                                >
                                                    <div className="text-sm font-semibold text-gray-900">{option.label}</div>
                                                    <div className="mt-1 text-xs leading-5 text-gray-500">{option.description}</div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </section>
                            </div>

                            <div className="mt-6 flex items-center justify-end gap-3">
                                <button
                                    type="button"
                                    onClick={onClose}
                                    className="rounded-2xl border border-gray-200 px-4 py-2.5 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50"
                                >
                                    Đóng
                                </button>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </Dialog>
        </Transition>
    );
};

export default ChatSettingsModal;
