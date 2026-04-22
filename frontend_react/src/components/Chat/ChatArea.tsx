import type { FC } from 'react';
import { useState, useRef, useEffect } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useAppStore } from '../../store/useAppStore';
import { Send, SlidersHorizontal, Trash2 } from 'lucide-react';
import { apiService } from '../../services/api';
import { detectQuestionLanguage } from '../../utils/language';
import CitationBadge from './CitationBadge';
import ChatSettingsModal from './ChatSettingsModal';
import ConfirmModal from '../Common/ConfirmModal';

type CitationDetail = {
    source_name: string;
    page?: number | null;
    snippet: string;
};

type MessageSegment =
    | { kind: 'text'; content: string }
    | { kind: 'citation'; number: number };

const markdownComponents: Components = {
    p: ({ children }) => <p className="mb-3 last:mb-0 leading-7 text-gray-800">{children}</p>,
    ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5 text-gray-800">{children}</ul>,
    ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5 text-gray-800">{children}</ol>,
    li: ({ children }) => <li className="leading-7">{children}</li>,
    a: ({ children, href }) => (
        <a href={href} target="_blank" rel="noreferrer" className="text-blue-600 underline underline-offset-2 hover:text-blue-700">
            {children}
        </a>
    ),
    blockquote: ({ children }) => (
        <blockquote className="mb-3 border-l-4 border-gray-200 pl-4 italic text-gray-600">
            {children}
        </blockquote>
    ),
    code: ({ children, className }) => {
        const isInline = !className;
        return isInline ? (
            <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[0.9em] text-gray-800">{children}</code>
        ) : (
            <code className={className}>{children}</code>
        );
    },
    pre: ({ children }) => (
        <pre className="mb-3 overflow-x-auto rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm text-gray-800">{children}</pre>
    ),
    table: ({ children }) => (
        <div className="mb-3 overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse text-left text-sm">{children}</table>
        </div>
    ),
    thead: ({ children }) => <thead className="bg-gray-50">{children}</thead>,
    th: ({ children }) => (
        <th className="border-b border-gray-200 px-3 py-2 font-semibold text-gray-900">{children}</th>
    ),
    td: ({ children }) => (
        <td className="border-b border-gray-100 px-3 py-2 align-top text-gray-700">{children}</td>
    ),
};

const inlineCitationMarkdownComponents: Components = {
    ...markdownComponents,
    p: ({ children }) => <span className="inline leading-7 text-gray-800">{children}</span>,
};

function splitContentByCitations(content: string): MessageSegment[] {
    const citationPattern = /\[(\d+)\]/g;
    const segments: MessageSegment[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = citationPattern.exec(content)) !== null) {
        if (match.index > lastIndex) {
            segments.push({ kind: 'text', content: content.slice(lastIndex, match.index) });
        }

        segments.push({ kind: 'citation', number: Number(match[1]) });
        lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
        segments.push({ kind: 'text', content: content.slice(lastIndex) });
    }

    return segments.length > 0 ? segments : [{ kind: 'text', content }];
}

function renderAssistantContent(content: string, citations: CitationDetail[] = []) {
    return splitContentByCitations(content).map((segment, index) => {
        if (segment.kind === 'citation') {
            const citation = citations[segment.number - 1];

            return citation ? (
                <CitationBadge
                    key={`citation-${segment.number}-${index}`}
                    number={segment.number}
                    sourceName={citation.source_name}
                    snippet={citation.snippet}
                    page={citation.page}
                />
            ) : (
                <span
                    key={`citation-${segment.number}-${index}`}
                    className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-gray-100 text-[10px] font-semibold text-gray-500"
                >
                    {segment.number}
                </span>
            );
        }

        return (
            <span key={`markdown-${index}`} className="inline align-baseline">
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={inlineCitationMarkdownComponents}
                >
                    {segment.content}
                </ReactMarkdown>
            </span>
        );
    });
}

export const ChatArea: FC = () => {
    const {
        chatMessages,
        addChatMessage,
        isTyping,
        setIsTyping,
        activeNotebookId,
        sources,
        hasActiveSources,
        chatSessionByNotebook,
        setChatSessionForNotebook,
        chatSettings,
        clearChatHistory,
        setError,
    } = useAppStore();

    const [inputValue, setInputValue] = useState('');
    const [textareaHeight, setTextareaHeight] = useState('44px');
    const [showClearConfirm, setShowClearConfirm] = useState(false);
    const [isClearingChat, setIsClearingChat] = useState(false);
    const [showChatSettings, setShowChatSettings] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const hasSources = sources.length > 0;
    const hasSelectedSources = hasActiveSources();
    const selectedSourceNames = sources
        .filter((source) => source.selected)
        .map((source) => source.title);
    const canChat = Boolean(activeNotebookId) && hasSelectedSources;
    const activeSessionId = activeNotebookId ? chatSessionByNotebook[activeNotebookId] : undefined;

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages, isTyping]);

    // Auto-resize textarea
    const handleTextareaChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
        setInputValue(e.target.value);

        // Reset height to auto to get actual scrollHeight
        e.target.style.height = 'auto';
        const height = Math.min(e.target.scrollHeight, 160);
        setTextareaHeight(`${height}px`);
        e.target.style.height = `${height}px`;
    };

    // Handle sending message
    const handleSendMessage = async () => {
        const question = inputValue.trim();
        if (!question || !activeNotebookId || !canChat) return;

        // Reset textarea
        setTextareaHeight('44px');

        // Add user message
        const userMessage = {
            id: `msg-${Date.now()}`,
            role: 'user' as const,
            content: question,
            timestamp: new Date().toISOString(),
        };

        addChatMessage(userMessage);
        setInputValue('');

        setIsTyping(true);

        try {
            setError(null);
            console.log('[ChatArea] Calling API with:', { question, activeSessionId });
            const result = await apiService.chatNotebook(activeNotebookId, {
                question,
                session_id: activeSessionId,
                source_names: selectedSourceNames.length > 0 ? selectedSourceNames : undefined,
                answer_language: detectQuestionLanguage(question),
                chatSettings,
            });

            console.log('[ChatArea] API response received:', {
                answer_length: result.answer?.length,
                answer_graph_length: result.answer_graph?.length,
                answer_graph_value: result.answer_graph,
                citations_count: result.citations?.length,
            });

            setChatSessionForNotebook(activeNotebookId, result.session_id);

            const hasGraphAnswer = result.answer_graph && result.answer_graph.trim().length > 0;
            console.log('[ChatArea] Has graph answer:', hasGraphAnswer, 'answer_graph:', result.answer_graph?.substring(0, 100));
            
            const content = hasGraphAnswer 
                ? `${result.answer}\n\n--- KẾT QUẢ GRAPH RAG ---\n\n${result.answer_graph}` 
                : result.answer;
            
            console.log('[ChatArea] Final message content length:', content.length);
            const aiMessage = {
                id: `msg-${Date.now()}`,
                role: 'assistant' as const,
                content,
                citations: result.citations.map((_, index) => index + 1),
                citationDetails: result.citations,
                timestamp: new Date().toISOString(),
            };

            addChatMessage(aiMessage);
        } catch (error) {
            const message =
                error instanceof Error && error.message.trim().length > 0
                    ? error.message
                    : 'Không thể gọi API chat. Vui lòng kiểm tra backend và thử lại.';

            addChatMessage({
                id: `msg-${Date.now()}-error`,
                role: 'assistant',
                content: message,
                timestamp: new Date().toISOString(),
            });
            setError(message);
        } finally {
            setIsTyping(false);
        }
    };

    // Handle Enter key
    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    // Focus textarea on mount
    useEffect(() => {
        textareaRef.current?.focus();
    }, []);

    const handleConfirmClearChat = async () => {
        if (!activeNotebookId || isClearingChat) return;

        try {
            setIsClearingChat(true);
            setError(null);
            await clearChatHistory(activeNotebookId);
            setShowClearConfirm(false);
        } catch (error) {
            const message =
                error instanceof Error && error.message.trim().length > 0
                    ? error.message
                    : 'Không thể xóa lịch sử trò chuyện.';
            setError(message);
        } finally {
            setIsClearingChat(false);
        }
    };

    return (
        <div className="flex flex-col h-full min-h-0 bg-white border-r border-gray-200">
            <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
                <div className="text-sm font-semibold text-gray-900">Hội thoại</div>
                <div className="panel-header-actions flex items-center gap-2">
                    <button
                        type="button"
                        onClick={() => setShowClearConfirm(true)}
                        disabled={!activeNotebookId || isTyping || chatMessages.length === 0}
                        className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-700 transition-colors hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        <Trash2 className="h-3.5 w-3.5 text-red-500" />
                        Xóa cuộc trò chuyện
                    </button>

                    <button
                        type="button"
                        onClick={() => setShowChatSettings(true)}
                        disabled={!activeNotebookId}
                        className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-700 transition-colors hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        <SlidersHorizontal className="h-3.5 w-3.5 text-gray-600" />
                        Cài đặt Chat
                    </button>
                </div>
            </div>

            {/* CHAT MESSAGES CONTAINER */}
            <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6 space-y-4">
                {chatMessages.length === 0 ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                            <h2 className="text-xl font-semibold text-gray-900 mb-2">
                                {hasSources ? (hasSelectedSources ? 'Bắt đầu hội thoại' : 'Chưa chọn nguồn tài liệu') : 'Chưa có nguồn dữ liệu'}
                            </h2>
                            <p className="text-sm text-gray-600">
                                {hasSources
                                    ? (hasSelectedSources
                                        ? 'Hỏi câu hỏi về các tài liệu của bạn'
                                        : 'Vui lòng chọn ít nhất 1 nguồn tài liệu để bắt đầu trò chuyện...')
                                    : 'Vui lòng thêm ít nhất một file nguồn ở cột bên trái để bắt đầu chat.'}
                            </p>
                        </div>
                    </div>
                ) : (
                    chatMessages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-lg shadow-sm ${msg.role === 'user'
                                    ? 'bg-blue-500 text-white rounded-br-none'
                                    : 'bg-gray-100 text-gray-900 rounded-bl-none'
                                    }`}
                            >
                                {msg.role === 'assistant' ? (
                                    <div className="prose prose-sm max-w-none text-gray-800 prose-headings:font-semibold prose-p:my-0 prose-table:my-3 prose-th:border prose-th:border-gray-200 prose-th:bg-gray-50 prose-th:px-3 prose-th:py-2 prose-td:border prose-td:border-gray-100 prose-td:px-3 prose-td:py-2">
                                        {renderAssistantContent(msg.content, msg.citationDetails || [])}
                                    </div>
                                ) : (
                                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                                )}
                            </div>
                        </div>
                    ))
                )}

                {/* TYPING INDICATOR */}
                {isTyping && (
                    <div className="flex justify-start">
                        <div className="bg-gray-100 text-gray-900 px-4 py-3 rounded-lg rounded-bl-none">
                            <div className="flex gap-1">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                    style={{ animationDelay: '0.1s' }}
                                />
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                    style={{ animationDelay: '0.2s' }}
                                />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* CHAT INPUT */}
            <div className="flex-shrink-0 border-t border-gray-200 p-4">
                {!hasSources && (
                    <p className="mb-3 text-xs font-medium text-amber-700">
                        Bạn cần thêm nguồn tài liệu trước khi đặt câu hỏi.
                    </p>
                )}
                {hasSources && !hasSelectedSources && (
                    <p className="mb-3 text-xs font-medium text-amber-700">
                        Vui lòng chọn ít nhất 1 nguồn tài liệu để bắt đầu trò chuyện...
                    </p>
                )}
                <div className="flex items-end gap-3">
                    <textarea
                        ref={textareaRef}
                        value={inputValue}
                        onChange={handleTextareaChange}
                        onKeyDown={handleKeyDown}
                        disabled={!canChat || isTyping}
                        placeholder={
                            hasSources
                                ? (hasSelectedSources
                                    ? 'Nhập câu hỏi...'
                                    : 'Vui lòng chọn ít nhất 1 nguồn tài liệu để bắt đầu trò chuyện...')
                                : 'Thêm nguồn tài liệu để bắt đầu chat'
                        }
                        style={{ height: textareaHeight }}
                        className="flex-1 px-4 py-2 bg-gray-100 border border-gray-300 rounded-lg resize-none text-sm text-gray-900 placeholder-gray-500 focus:outline-none focus:bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 max-h-40 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-400"
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!inputValue.trim() || !canChat || isTyping}
                        className="flex-shrink-0 p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white rounded-lg transition-colors"
                        title="Gửi (Enter)"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </div>

            <ConfirmModal
                open={showClearConfirm}
                title="Xóa cuộc trò chuyện"
                description="Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện của sổ ghi chú này không?"
                confirmText="Xóa"
                isSubmitting={isClearingChat}
                onConfirm={handleConfirmClearChat}
                onClose={() => {
                    if (!isClearingChat) {
                        setShowClearConfirm(false);
                    }
                }}
            />

            <ChatSettingsModal open={showChatSettings} onClose={() => setShowChatSettings(false)} />
        </div>
    );
};

export default ChatArea;
