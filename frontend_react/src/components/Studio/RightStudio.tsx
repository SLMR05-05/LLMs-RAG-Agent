import type { FC } from 'react';
import { useAppStore } from '../../store/useAppStore';
import {
    PanelRight,
    PanelRightOpen,
    List,
    HelpCircle,
    BookOpen,
    Settings,
} from 'lucide-react';

interface RightStudioProps {
    collapsed: boolean;
}

export const RightStudio: FC<RightStudioProps> = ({ collapsed }) => {
    const studioMode = useAppStore((s) => s.studioMode);
    const setStudioMode = useAppStore((s) => s.setStudioMode);
    const toggleRightPanel = useAppStore((s) => s.toggleRightPanel);

    const studioModes = [
        { id: 'outline', label: 'Dàn bài', icon: List },
        { id: 'faq', label: 'Câu hỏi phổ biến', icon: HelpCircle },
        { id: 'briefing', label: 'Tóm tắt', icon: BookOpen },
        { id: 'custom', label: 'Tùy chỉnh', icon: Settings },
    ] as const;

    return (
        <aside
            className={`flex flex-col h-full min-h-0 bg-white border-l border-gray-200 transition-all duration-300 ease-out overflow-hidden ${collapsed ? 'w-12' : 'w-[310px]'
                }`}
        >
            {/* HEADER */}
            <div className="flex items-center justify-between flex-shrink-0 px-4 py-3.5 border-b border-gray-200">
                {!collapsed && (
                    <h2 className="text-sm font-semibold text-gray-900">Studio</h2>
                )}
                <button
                    onClick={toggleRightPanel}
                    className="ml-auto p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title={collapsed ? 'Mở rộng' : 'Thu gọn'}
                >
                    {collapsed ? (
                        <PanelRightOpen className="w-4 h-4" />
                    ) : (
                        <PanelRight className="w-4 h-4" />
                    )}
                </button>
            </div>

            {!collapsed && (
                <>
                    {/* MODE TABS */}
                    <div className="flex border-b border-gray-200">
                        {studioModes.map(({ id, label, icon: Icon }) => (
                            <button
                                key={id}
                                onClick={() =>
                                    setStudioMode(id as 'outline' | 'faq' | 'briefing' | 'custom')
                                }
                                className={`flex-1 px-3 py-3 text-xs font-medium border-b-2 transition-colors duration-200 flex items-center justify-center gap-1.5 ${studioMode === id
                                    ? 'border-blue-500 text-blue-600 bg-blue-50'
                                    : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                    }`}
                                title={label}
                            >
                                <Icon className="w-4 h-4" />
                                <span className="hidden sm:inline">{label}</span>
                            </button>
                        ))}
                    </div>

                    {/* CONTENT AREA */}
                    <div className="flex-1 overflow-y-auto p-4">
                        {studioMode === 'outline' && (
                            <div>
                                <h3 className="text-sm font-semibold text-gray-900 mb-3">
                                    Dàn bài
                                </h3>
                                <ul className="space-y-2 text-xs text-gray-600">
                                    <li className="flex items-start gap-2">
                                        <span className="text-blue-500 mt-1">•</span>
                                        <span>Giới thiệu về hệ thống</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-blue-500 mt-1">•</span>
                                        <span>Kiến trúc cơ sở dữ liệu</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-blue-500 mt-1">•</span>
                                        <span>Quy trình xét tuyển</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-blue-500 mt-1">•</span>
                                        <span>Yêu cầu bảo mật</span>
                                    </li>
                                </ul>
                            </div>
                        )}

                        {studioMode === 'faq' && (
                            <div>
                                <h3 className="text-sm font-semibold text-gray-900 mb-3">
                                    Câu hỏi phổ biến
                                </h3>
                                <div className="space-y-3 text-xs text-gray-600">
                                    <div>
                                        <p className="font-medium text-gray-900 mb-1">
                                            Cơ sở dữ liệu tàng?
                                        </p>
                                        <p>PostgreSQL với phân vùng dữ liệu theo năm học</p>
                                    </div>
                                    <div>
                                        <p className="font-medium text-gray-900 mb-1">
                                            Độ trễ phản hồi?
                                        </p>
                                        <p>Dưới 2 giây cho 4.000 người dùng đồng thời</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {studioMode === 'briefing' && (
                            <div>
                                <h3 className="text-sm font-semibold text-gray-900 mb-3">
                                    Tóm tắt
                                </h3>
                                <p className="text-xs text-gray-600 leading-relaxed">
                                    Hệ thống Quản lý Xét Tuyển là một nền tảng xử lý dữ liệu thí sinh với khả năng xử lý tối đa 4.000 người dùng đồng thời, đáp ứng các yêu cầu bảo mật cao nhất.
                                </p>
                            </div>
                        )}

                        {studioMode === 'custom' && (
                            <div>
                                <h3 className="text-sm font-semibold text-gray-900 mb-3">
                                    Tùy chỉnh
                                </h3>
                                <p className="text-xs text-gray-500">
                                    Không có nội dung tùy chỉnh. Hãy đặt câu hỏi để tạo nội dung.
                                </p>
                            </div>
                        )}
                    </div>
                </>
            )}
        </aside>
    );
};

export default RightStudio;
